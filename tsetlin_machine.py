import numpy as np
import random
import torch
from torch import IntTensor, ByteTensor


class TsetlinMachine:
    """The Tsetlin Machine.

    The learned state variables in this model are Tsetlin automata. An automata
    consists of an integer counter, a boolean 'action' (which tells if the
    counter is in the upper half of its counting range), and increment /
    decrement operations. The counting range is clamped to the range
    [0, 2 * states), where 'states' is a hyperparameter.

    The number of automata depends on:
        (1) The number of boolean input features, and
        (2) The number of clauses in the machine.

    Each clause requires 2 arrays of automata, each array having length equal to
    the number of inputs. One array manages the conjunction of non-inverting
    inputs, the other manages the conjunction of inverting inputs. So the total
    number of automata in the machine is:

        2 * #clauses * #inputs

    and is represented by an 'automata' matrix, a 2D tensor of type int,
    indexed by [clause, input].

    Clauses are divided into two "polarities," positive clauses and negative
    clauses. The first half of the automate matrix represents positive clauses,
    the second half negative clauses. Each of those halves is also divided in
    half to separate automata controlling non-inverting inputs from those
    controlling inverting inputs. The division looks like this:

        automata:
            +-------------------------------------------+
            |  positive polarity, non-inverting inputs  |
            +-------------------------------------------+
            |  negative polarity, non-inverting inputs  |
            +-------------------------------------------+

        inverting_automata:
            +-------------------------------------------+
            |  positive polarity, inverting inputs      |
            +-------------------------------------------+
            |  negative polarity, inverting inputs      |
            +-------------------------------------------+

    Attributes:
        class_count: Number of boolean outputs (classes).
        clause_count: Total number of clauses in the machine.
        feature_count: Number of boolean inputs.
        state_count: Number of states in each Tsetlin automata.
        s: system parameter (?)
        threshold: system parameter (?)
        automata: 2D tensor of Tsetlin automata controlling clauses.
        inverting_automata: 2D tensor of Tsetlin automata controlling clauses.

    """
    def __init__(self, class_count: int, clause_count: int, feature_count: int,
                 state_count: int, s, threshold):
        print('creating machine...')
        # The clauses are divided equally between the classes.
        if (clause_count // class_count) * class_count != clause_count:
            raise ValueError('clause_count must be a multiple of class_count')
        self.class_count = class_count
        self.clause_count = clause_count
        self.feature_count = feature_count
        self.state_count = state_count
        self.s = s
        self.threshold = threshold

        # Each automata in the automata array is randomly initialized to either
        # state_count or state_count + 1
        shape = (clause_count, feature_count)
        initial_state = np.random.randint(state_count, state_count + 2, shape)
        self.automata = torch.from_numpy(initial_state)
        initial_state = np.random.randint(state_count, state_count + 2, shape)
        self.inverting_automata = torch.from_numpy(initial_state)

        self.action = self.automata > self.state_count
        self.inverting_action = self.inverting_automata > self.state_count
        print('...done')

    def update_action(self):
        """Update the actions from the automata, needed after learning."""
        self.action = self.automata > self.state_count
        self.inverting_action = self.inverting_automata > self.state_count

    def evaluate_clauses(self, input: torch.ByteTensor) -> torch.ByteTensor:
        """Evaluate all clauses in the array.

        Args:
            input: 1D boolean array (length = feature_count) holding the input
                vector to the machine.

        Returns:
            1D boolean array of the outputs of each clause. The first half
                contains the positive polarity clauses, the second half contains
                the negative polarity clauses.
        """
        # First we process the non-inverting automata.
        # We collect the 'used_bits' matrix, those bits that are used by each
        # clause in computing the conjunction of the non-inverted input.s
        # Each row of 'used_bits' are the non-inverted used bits for one clause.
        used_bits = self.action

        # For each clause, we mask out input bits which are not used. If the
        # number of remaining bits equals the number of bits in used_bits for
        # that clause, then the conjunction of the non-inverting bits is True.
        masked_input = used_bits & input.expand_as(used_bits)
        used_row_sums = torch.sum(used_bits.int(), 1)
        masked_input_row_sums = torch.sum(masked_input.int(), 1)
        conjunction = used_row_sums.eq(masked_input_row_sums)

        # Repeat the above computations for the inverting automata.
        inv_input = ~input
        inv_used_bits = self.inverting_action
        inv_masked_input = inv_used_bits & inv_input.expand_as(inv_used_bits)
        inv_used_row_sums = torch.sum(inv_used_bits.int(), 1)
        inv_masked_input_row_sums = torch.sum(inv_masked_input.int(), 1)
        inv_conjunction = inv_used_row_sums.eq(inv_masked_input_row_sums)

        # The final output of each clause is the conjunction of:
        #   (1) conjunction of used, non-inverting inputs
        #   (2) conjunction of used, inverted inputs
        clause_result = conjunction & inv_conjunction
        return clause_result

    def sum_up_class_votes(self, clause_outputs: ByteTensor) -> IntTensor:
        """Add up votes for all classes.

        Args:
            clause_outputs: 1D boolean array of the outputs of each clause.
                The first half contains the positive polarity clauses, the
                second half contains the negative polarity clauses.

        Returns:
            1D tensor with vote count for each class.

        """
        # We split the clauses into positive polarity and negative polarity,
        # then compute the polarity-weighted votes.
        clauses = clause_outputs.shape[0]
        positive = clause_outputs[0 : clauses // 2]
        negative = clause_outputs[clauses//2 :]
        votes = positive - negative

        # The votes are spread evenly across the classes.
        votes_per_class = votes.shape[0] // self.class_count
        class_votes = []
        offset = 0
        for c in range(self.class_count):
            subvotes = votes[offset : offset + votes_per_class]
            sum = torch.sum(subvotes)

            # Not clear how the following block helps
            if sum > self.threshold:
                sum = self.threshold
            elif sum < -self.threshold:
                sum = -self.threshold

            class_votes.append(sum)
        return IntTensor(class_votes)

    def predict(self, input: ByteTensor) -> IntTensor:
        """Forward inference of input.

        Args:
            input: 1D boolean input.

        Returns:
            The index of the class of the input (scalar held in tensor).
        """
        clause_outputs = self.evaluate_clauses(input)
        class_votes = self.sum_up_class_votes(clause_outputs)
        value, index = torch.max(class_votes, 0)
        return index

    def evaluate(self, inputs: ByteTensor, targets: IntTensor) -> float:
        """Evaluate the machine on a dataset.

        Args:
            inputs: 2D array of inputs, each row is one (boolean) input vector.
            targets: 1D array of class indices, one for each input.

        Returns:
            Classification accuracy of the machine.
        """
        errors = 0
        examples = targets.shape[0]
        for i in range(examples):
            if i % 100 == 0:
                print('.', end='', flush=True)
            input = torch.from_numpy(inputs[i]).byte()
            target = np.zeros((1, ), dtype=np.int32)
            target[0] = targets[i]
            target = torch.from_numpy(target)
            prediction = self.predict(input).int()
            if not prediction.equal(target):
                errors += 1
        accuracy = (examples - errors) / examples
        return accuracy

    def train(self, input: ByteTensor, target: int):
        """Train the machine with a single example.

        Args:
            input: 1D array of booleans.
            target: The class of the input

        """
        # Randomly pick one of the other classes for pairwise learning.
        negative_target_class = target
        while negative_target_class == target:
            negative_target_class = random.randint(0, self.class_count)

        ###############################
        ### Calculate Clause Output ###
        ###############################
        clause_outputs = self.evaluate_clauses(input)

        ###########################
        ### Sum up Clause Votes ###
        ###########################
        votes = self.sum_up_class_votes(clause_outputs)

        #####################################
        ### Calculate Feedback to Clauses ###
        #####################################

        # Initialize feedback to clauses
        feedback = IntTensor((self.clause_count, )).zero_()

        # Calculate feedback to clauses

        # Feedback to target class


        #################################
        ### Train Individual Automata ###
        #################################

        ####################################################
        ### Type I Feedback (Combats False Negatives) ###
        ####################################################

        #####################################################
        ### Type II Feedback (Combats False Positives) ###
        #####################################################

if __name__ == '__main__':
    print('Start')
    from mnist_demo import mnist_dataset
    print('creating mnist...')
    X, y = mnist_dataset(training=False)
    print('...done')
    machine = TsetlinMachine(class_count=10, clause_count=200,
                             feature_count=28*28, state_count=200,
                             s=3.9, threshold=15)
    accuracy = machine.evaluate(X, y)
    print('accuracy:', accuracy)



class Automata:
    """A 1D array of Tsetlin automata.

    Attributes:
        size: The number of automata in the array.
        state_count: Number of states in each automata.
        current_state (array of bool): The current state of each automata
            in the array.

    """
    def __init__(self, size: int, state_count: int):
        self.size = size
        self.state_count = state_count

        # Each automata in the array is randomly initialized to either
        # state_count or state_count + 1
        self.current_state = np.random.randint(state_count, state_count+2, size)

    def action(self) -> np.ndarray:
        """Get the action for each automata in the array:

        Return: Array of boolean, True if an automata counter is above the
            half-way mark, False if below.
        """
        return self.current_state > self.state_count


class ConjunctiveClause:
    """Conjunctive clause, defined on p. 8 of the paper.

    Attributes:
        size: Number of input features to the clause.
        used: Boolean flags for non-inverted inputs used in conjunction.
        used_inverted: Boolean flags for non-inverted inputs used.
        auto_used: Automata controlling used flags.
        auto_used_inverted: Automata controlling used_inverted flags.

    """
    def __init__(self, size: int, state_count: int):
        self.size = size
        self.used = np.full((size, ), False)
        self.used_inverted = np.full((size, ), False)
        self.auto_used = Automata(size, state_count)
        self.auto_used_inverted = Automata(size, state_count)

    def evaluate(self, input: np.ndarray) -> bool:
        """Evalute the output of the clause for the given input."""
        input_inverted = ~input
        return np.array_equal(input & self.used, self.used) and \
               np.array_equal(input_inverted & self.used_inverted,
                              self.used_inverted)

    def update_used_flags(self):
        """Update used flags from the automata actions."""
        self.used = self.auto_used.action()
        self.used_inverted = self.auto_used_inverted.action()


class Output:
    """A submachine sourcing a single output of the entire machine.

    Attributes:
        clauses: The clauses used by the submachine.
        automata: The automata teams supervising the clauses.

    """
    def __init__(self, num_inputs: int, num_clauses: int, num_states: int):
        self.clauses = [ConjunctiveClause(num_inputs, num_states)
                        for _ in range(num_clauses)]

    def evaluate(self, input: np.ndarray) -> int:
        """Evaluate the output for the given input."""
        sum = 0
        for index, clause in enumerate(self.clauses):
            clause_output = 1 if clause.evaluate(input) else 0
            if index % 2 == 1:
                sum += clause_output   # Odd clause indices: positive polarity.
            else:
                sum -= clause_output   # Even clause indices: negative polarity.
        return sum


class TsetlinMachineOld:
    """A Tsetlin machine with multiple inputs and outputs, all boolean.

    Attributes:
        class_count: Number of boolean outputs (classes).
        clause_count: Total number of clauses in the machine.
        feature_count: Number of boolean inputs.
        state_count: Number of states in each Tsetlin automata.
        s: system parameter (?)
        threshold: system parameter (?)
        outputs (List[Output]): The submachines, one for each output.

    """

    def __init__(self, class_count: int, clause_count: int, feature_count: int,
                 state_count: int, s, threshold):
        self.class_count = class_count
        self.clause_count = clause_count
        self.feature_count = feature_count
        self.state_count = state_count
        self.s = s
        self.threshold = threshold
        clauses_per_output = clause_count // class_count
        if clauses_per_output * class_count != clause_count:
            raise ValueError('number of clauses must be a multiple of classes')
        self.outputs = [Output(feature_count, clauses_per_output, state_count)
                        for _ in range(class_count)]

    def predict(self, input: np.ndarray) -> np.ndarray:
        """Return the machine output as an array of bool."""
        result = np.full((len(self.outputs),), False)
        for i in range(result.shape[0]):
            result[i] = self.outputs[i].evaluate(input)
        return result

    def evaluate(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        """Evaluate the performance of the machine on a test set.

        Args:
            inputs: Boolean matrix of input examples, each row is one input.
            targets: For each row, contains the correct class number for that
                row.

        Returns:
             Fraction of inputs that were correctly classified by the machine.
        """
        correct = 0
        samples = len(targets)
        for sample in range(samples):
            input = inputs[sample]
            target = targets[sample]
            results = [self.outputs[i].evaluate(input)
                       for i in range(self.class_count)]
            biggest_output = max(results)
            if results[target] == biggest_output:
                correct += 1
        return correct / samples

'''
from torchvision import datasets

def mnist_dataset(training=True) -> (np.ndarray, np.ndarray):
    dataset = datasets.MNIST('./data', train=training, download=True)
    rows = len(dataset)
    cols = 28 * 28
    X = np.zeros((rows, cols), dtype=np.int32)
    y = np.zeros((rows, ), dtype=np.int32)
    print(X.shape)
    for i in range(rows):
        image, target = dataset[i]
        X[i] = (np.array(image).flatten() // 128).astype(np.int32)
        y[i] = target
    return X, y


# Parameters for the Tsetlin Machine
T = 15
s = 3.9
number_of_clauses = 1600
states = 200

# Parameters of the pattern recognition problem
number_of_features = 28 * 28
number_of_classes = 10
machine = TsetlinMachine(number_of_classes, number_of_clauses,
                         number_of_features, states, s, T)
inputs, targets = mnist_dataset(training=False)
result = machine.evaluate(inputs, targets)
print('accuracy:', result)
'''