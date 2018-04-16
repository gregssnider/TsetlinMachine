import numpy as np
import torch
from torch import IntTensor, ByteTensor, CharTensor
import time
import sys


class TsetlinMachine2:
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
    clauses. The first half of an array of clauses are "positive", the second
    half "negative". Those subarrays are further subdivided into clauses for
    each of the classes:

        clauses:
            +-------------------------------------------+
            |  pos polarity       class 0               |
            |..                                       ..|
            |                     class 1               |
            |..                                       ..|
            |                     class 2               |
            |..                                       ..|
            |                     class 3               |
            +-------------------------------------------+
            |  neg polarity       class 0               |
            |..                                       ..|
            |                     class 1               |
            |..                                       ..|
            |                     class 2               |
            |..                                       ..|
            |                     class 3               |
            +-------------------------------------------+

    The automata matrix has the same structure as an array of clauses, with
    the first half representing positive polarity, the second half representing
    negative polarity. These are subdivided into clauses per class as above.
    The number of columns is equal to the number of input features, so the
    total number of automata is number_of_clauses * number_of_features

        automata:
            +----------------------------------------------+
            |  pos polarity clauses, non-inverting inputs  |
            |                                              |
            |                                              |
            +----------------------------------------------+
            |  neg polarity clauses, non-inverting inputs  |
            |                                              |
            |                                              |
            +----------------------------------------------+


    The inverting automata matrix has the same structure:

        inverting_automata:
            +----------------------------------------------+
            |  pos polarity clauses, non-inverting inputs  |
            |                                              |
            |                                              |
            +----------------------------------------------+
            |  neg polarity clauses, non-inverting inputs  |
            |                                              |
            |                                              |
            +----------------------------------------------+


    Attributes:
        class_count: Number of boolean outputs (classes).
        clause_count: Total number of clauses in the machine.
        clauses_per_class: Number of clauses for each class, must be mult. of 2.
        feature_count: Number of boolean inputs.
        state_count: Number of states in each Tsetlin automata.
        s: system parameter (?)
        threshold: system parameter (?)
        automata: 2D tensor of Tsetlin automata controlling clauses.
        inverting_automata: 2D tensor of Tsetlin automata controlling clauses.

    """
    def __init__(self, class_count: int, clause_count: int, feature_count: int,
                 states: int, s, threshold):
        if clause_count % (2 * class_count) != 0:
            raise ValueError("# clauses must be a multiple of (2 * # classes)")

        self.class_count = class_count
        self.clause_count = clause_count
        self.feature_count = feature_count
        self.states = states
        self.s = s
        self.threshold = threshold

        self.clauses_per_class = clause_count // class_count

        # Automata and action tensors are 4D, indexed by:
        #    polarity   (0 => positive, 1 => negative)
        #    class index
        #    class_clauses per polarity
        #    input feature index
        #
        polarities = 2
        clause_shape = (polarities, class_count,
                        self.clauses_per_class // polarities, feature_count)

        self.automata = IntTensor(*clause_shape).random_(states, states + 2)
        self.inv_automata = IntTensor(*clause_shape).random_(states, states + 2)
        self.action = IntTensor(*clause_shape)
        self.inv_action = IntTensor(*clause_shape)
        self.update_action()

        assert isinstance(self.automata, IntTensor), type(self.automata)
        assert isinstance(self.inv_automata, IntTensor)

    def __str__(self):
        string = 'TsetlinMachine2\n'
        string += '  automata\n'
        string += str(self.automata)
        string += '  inv_automata\n'
        string += str(self.inv_automata)
        string += '  action\n'
        string += str(self.action)
        string += '  inv_action\n'
        string += str(self.inv_action)
        return string


    def update_action(self):
        """Update the actions from the automata, needed after learning."""
        self.action = self.automata > self.states
        self.inv_action = self.inv_automata > self.states

    def evaluate_clauses(self, input: ByteTensor) -> ByteTensor:
        """Evaluate all clauses in the array.

        Args:
            input: 1D boolean array (length = feature_count) holding the input
                vector to the machine.

        Returns:
            1D boolean array of the outputs of each clause. The first half
                contains the positive polarity clauses, the second half contains
                the negative polarity clauses.
        """
        assert isinstance(input, ByteTensor)
        assert input.shape == (self.feature_count, )

        # First we process the non-inverting automata.
        # We collect the 'used_bits' matrix, those bits that are used by each
        # clause in computing the conjunction of the non-inverted input.s
        # Each row of 'used_bits' are the non-inverted used bits for one clause.
        used_bits = self.action.view(-1, self.feature_count)

        # For each clause, we mask out input bits which are not used. If the
        # number of remaining bits equals the number of bits in used_bits for
        # that clause, then the conjunction of the non-inverting bits is True.
        masked_input = used_bits & input.expand_as(used_bits)
        used_row_sums = torch.sum(used_bits.int(), 1)
        masked_input_row_sums = torch.sum(masked_input.int(), 1)
        conjunction = used_row_sums.eq(masked_input_row_sums)
        assert type(conjunction) == ByteTensor, str(type(conjunction))

        # Repeat the above computations for the inverting automata.
        inv_input = ~input
        inv_used_bits = self.inv_action.view(-1, self.feature_count)
        inv_masked_input = inv_used_bits & inv_input.expand_as(inv_used_bits)
        inv_used_row_sums = torch.sum(inv_used_bits.int(), 1)
        inv_masked_input_row_sums = torch.sum(inv_masked_input.int(), 1)
        inv_conjunction = inv_used_row_sums.eq(inv_masked_input_row_sums)

        # The final output of each clause is the conjunction of:
        #   (1) conjunction of used, non-inverting inputs
        #   (2) conjunction of used, inverted inputs
        clause_result = conjunction & inv_conjunction
        assert isinstance(clause_result, ByteTensor), str(type(clause_result))
        return clause_result

    def sum_up_class_votes(self, clause_outputs: ByteTensor) -> IntTensor:
        """Add up votes for all classes.

        This is where we structure the clause outputs (which are unaware of
        classes and clause polarities).

        Args:
            clause_outputs: shape = (self.clause_count, ), boolean output of
                each clause.

        Returns:
            shape = (self.class_count, ), integer sum of votes for each class.

        """
        assert isinstance(clause_outputs, ByteTensor)
        assert clause_outputs.shape == (self.clause_count, )


        ##### (polarity, class, clause_in_half_class, feature)

        # We split the clauses into positive polarity and negative polarity,
        # then compute the polarity-weighted votes.
        #
        # clause_outputs.shape = (2, classes, clauses_per_class // 2)
        clause_outputs = clause_outputs.view(2, self.class_count, -1)
        positive = clause_outputs[0].int()
        negative = clause_outputs[1].int()
        votes = positive - negative   # shape = (classes, clauses_per_class // 2)

        # The votes are spread evenly across the classes.
        class_votes = torch.sum(votes, dim=1)

        ########################################## Do we need this clamp ?????????????
        class_votes = torch.clamp(class_votes, -self.threshold, self.threshold)
        assert class_votes.shape == (self.class_count, )
        return class_votes

    def predict(self, input: ByteTensor) -> int:
        """Forward inference of input.

        Args:
            input: 1D boolean input.

        Returns:
            The index of the class of the input (scalar held in tensor).
        """
        assert isinstance(input, ByteTensor)
        assert input.shape == (self.feature_count, )

        clause_outputs = self.evaluate_clauses(input)
        class_votes = self.sum_up_class_votes(clause_outputs)
        value, index = torch.max(class_votes, 0)
        return index[0]

    def evaluate(self, inputs: np.ndarray, targets: np.ndarray, notused) -> float:
        """Evaluate the machine on a dataset.

        Args:
            inputs: 2D array of inputs, each row is one (boolean) input vector.
            targets: 1D array of class indices, one for each input.

        Returns:
            Classification accuracy of the machine.
        """
        inputs = torch.from_numpy(inputs.astype(np.uint8)).byte()
        targets = torch.from_numpy(targets.astype(np.uint8)).int()
        assert isinstance(inputs, ByteTensor)
        assert inputs.shape[1] == self.feature_count
        assert isinstance(targets, IntTensor)
        assert targets.shape == (inputs.shape[0], )

        errors = 0
        examples = targets.shape[0]
        for i in range(examples):
            if i % 100 == 0:
                print('.', end='', flush=True)
            input = inputs[i]
            target = targets[i]
            prediction = self.predict(input)
            if prediction != target:
                errors += 1
        accuracy = (examples - errors) / examples
        return accuracy

    def _low_probability(self, rows: int, columns: int) -> ByteTensor:
        """Compute an array of low probabilities.

        Each element in the array is 1 with probability (1 / s).

        Returns:
            boolean array of shape [rows][columns]
        """
        return torch.rand((rows, columns)) <= 1.0 / self.s

    def _high_probability(self, rows: int, columns: int) -> ByteTensor:
        """Compute an array of high probabilities.

        Each element in the array is 1 with probability (s-1 / s).

        Returns:
            boolean array of shape [rows][columns]
        """
        return torch.rand((rows, columns)) <= (self.s - 1.0) / self.s


    def train(self, input: ByteTensor, target_class: int):
        """Train the machine with a single example.

        Called 'update' in original code

        Args:
            input: 1D array of booleans.
            target: The class of the input

        """
        assert isinstance(input, ByteTensor)
        assert input.shape == (self.feature_count, )
        assert target_class >= 0 and target_class < self.class_count

        # Randomly pick one of the other classes for pairwise learning.
        anti_target_class = target_class
        while anti_target_class == target_class:
            anti_target_class = np.random.randint(0, self.class_count)

        assert anti_target_class < self.class_count

        ###############################
        ### Calculate Clause Output ###
        ###############################
        clause_outputs = self.evaluate_clauses(input)
        assert clause_outputs.shape == (self.clause_count, )

        ###########################
        ### Sum up Clause Votes ###
        ###########################
        votes = self.sum_up_class_votes(clause_outputs)
        assert votes.shape == (self.class_count, )

        # Automata and action tensors are 2D, indexed by:
        #    polarity   (0 => positive, 1 => negative)
        #    class index
        #    class_clauses per polarity
        #    input feature index
        #


        #####################################
        ### Calculate Feedback to Clauses ###
        #####################################

        # Process target -- random selection of target clauses to update
        thresh = (1.0 / (self.threshold * 2)) * (self.threshold - votes[target_class])
        target_feedback = (torch.rand((2, self.clauses_per_class // 2)) <= thresh).int()

        # Process anti-target -- random selection of anti target clauses to update
        thresh = (1.0 / (self.threshold * 2)) * (self.threshold + votes[target_class])
        anti_target_feedback = (torch.rand((2, self.clauses_per_class // 2)) <= thresh).int()

        #################################
        ### Train Individual Automata ###
        #################################
        low_prob = self._low_probability(self.clause_count, self.feature_count)
        high_prob = self._high_probability(self.clause_count, self.feature_count)


        # The reshape trick allows us to multiply the rows of a 2D matrix,
        # with the rows of the 1D clause_output.
        clause_matrix = clause_outputs.view(-1, 1)
        inv_clause_matrix = clause_matrix ^ 1
        feedback_matrix = feedback_to_clauses.view(-1, 1)
        pos_feedback_matrix = (feedback_matrix > 0)
        neg_feedback_matrix = (feedback_matrix < 0)

        assert low_prob.shape == (self.clause_count, self.feature_count)
        assert clause_matrix.shape == (self.clause_count, 1)

        # Vectorization -- this is essentially unreadable. It replaces
        # the commented out code just below it
        low_delta = inv_clause_matrix * (-1 * low_prob)
        delta =  clause_matrix * (input * high_prob - (1-input) * low_prob)
        delta_neg = clause_matrix * (-1 * input * low_prob + (1 - input) * high_prob)

        not_action_include = self.automata <= self.states
        not_action_include_negated = self.inv_automata <= self.states

        self.automata += (pos_feedback_matrix * (low_delta + delta) + \
            neg_feedback_matrix * (clause_matrix * (1 - input) * (not_action_include))).int()

        self.inv_automata += (pos_feedback_matrix * (low_delta + delta_neg) + \
            neg_feedback_matrix * clause_matrix * input * (not_action_include_negated)).int()

        self.automata.clamp(1, 2 * self.states)
        self.inv_automata.clamp(1, 2 * self.states)
        self.update_action()

    def fit(self, X: np.ndarray, y: np.ndarray, number_of_examples, epochs=100):
        """Train the network.

        Args:
            X: Matrix of inputs, one input per row.
            y: Vector of categories, one per row of inputs.
            number_of_examples: Rows in X and y.
            epochs: Number of training epochs.
        """
        assert len(X.shape) == 2 and len(y.shape) == 1
        assert X.shape[0] == y.shape[0] == number_of_examples
        assert X.shape[1] == self.feature_count

        # Convert input arrays to tensors.
        X = torch.from_numpy(X.astype(np.uint8))
        y = torch.from_numpy(y.astype(np.uint8))

        assert isinstance(X, ByteTensor)
        assert isinstance(y, ByteTensor)

        random_index = np.arange(number_of_examples)
        print()
        for epoch in range(epochs):
            print('\r epoch', epoch, end='', flush=True)
            np.random.shuffle(random_index)
            for i in range(number_of_examples):
                example_id = random_index[i]
                target_class = y[example_id]
                Xi = X[example_id]
                self.train(Xi, target_class)
        return


if __name__ == '__main__':
    # Parameters for the Tsetlin Machine
    T = 15
    s = 3.9
    number_of_clauses = 20
    states = 100

    # Parameters of the pattern recognition problem
    number_of_features = 12
    number_of_classes = 2

    # Training configuration
    epochs = 200

    # Loading of training and test data
    training_data = np.loadtxt("NoisyXORTrainingData.txt").astype(dtype=np.int8)
    test_data = np.loadtxt("NoisyXORTestData.txt").astype(dtype=np.int8)
    X_training = training_data[:, 0:12]  # Input features
    y_training = training_data[:, 12]  # Target value
    X_test = test_data[:, 0:12]  # Input features
    y_test = test_data[:, 12]  # Target value

    print("Noisy XOR")
    sum_accuracy = 0
    steps = 50
    for step in range(steps):
        start_time = time.time()
        '''
        tsetlin_machine = MultiClassTsetlinMachine(
            number_of_classes, number_of_clauses, number_of_features, states, s, T)
        '''
        tsetlin_machine = TsetlinMachine2(
            number_of_classes, number_of_clauses, number_of_features, states, s, T)
        tsetlin_machine.fit(X_training, y_training, y_training.shape[0], epochs)
        elapsed_time = time.time() - start_time
        accuracy = tsetlin_machine.evaluate(X_test, y_test, y_test.shape[0])
        print("  ", step," Accuracy on test data (no noise):", accuracy,
              ', elapsed time:', elapsed_time)
        sum_accuracy += accuracy
    print('Avg accuracy', sum_accuracy / steps)

