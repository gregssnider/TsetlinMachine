import numpy as np
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
        initial_state = np.random.randint(state_count, state_count + 2,
                                          (2 * clause_count, feature_count))
        self.automata = torch.from_numpy(initial_state)

    def action(self, automata_tensor: IntTensor) -> ByteTensor:
        """Compute the action of the given automata tensor.

        Returns:
            Boolean matrix with actions (True or False) for each autonoma.
        """
        return automata_tensor > self.state_count

    def evaluate_clauses(self, input: torch.ByteTensor) -> torch.ByteTensor:
        """Evaluate all clauses in the array.

        Args:
            input: 1D boolean array of length feature count holding the input
                vector to the machine.

        Returns:
            1D boolean array of the outputs of each clause. The first half
                contains the positive polarity clauses, the second half contains
                the negative polarity clauses.
        """

        action = self.action(self.automata)
        input = input.expand_as(action)
        used = action & input

        conjunction = used.eq(input * used)


        action_inv = self.action(self.automata_inv)
        input_inv = (~input).expand_as(action_inv)
        used_inv = action_inv & input_inv




        action_inverted = self.action(self.inverting_automata)
        used = action & input  # relying on broadcasting here
        used_inverted = action_inverted & ~input

        selected = torch.mv(self.non_inverting_automata(), input)
        selected_inverted = torch.mv(self.inverting_automata(), ~input)


        input_inverted = ~input
        return np.array_equal(input & self.used, self.used) and \
               np.array_equal(input_inverted & self.used_inverted,
                              self.used_inverted)

if __name__ == '__main__':
    def check_constructor():
        machine = TsetlinMachine(class_count=10, clause_count=300,
                                 feature_count=28*28, state_count=100,
                                 s=3.9, threshold=15)
        assert machine.automata.shape == \
               (2 * machine.clause_count, machine.feature_count)

    print('Testing TsetlinMachine...')
    check_constructor()

    t1 = torch.Tensor([
        [1, 2],
        [3, 4],
        [5, 6]
    ])

    t2 = torch.Tensor([
        [7, 8],
    ])

    print('t1')
    print(t1)
    print('t2')
    print(t2)
    print('t2.expand_as(t1)')
    print(t2.expand_as(t1))


    print('...passed')



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