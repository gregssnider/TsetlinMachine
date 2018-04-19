import numpy as np
import time
import torch
use_cuda = False  # torch.cuda.is_available()
if use_cuda:
    print('using GPU (CUDA)')
    from torch.cuda import IntTensor, ByteTensor, CharTensor, FloatTensor
else:
    print('using CPU')
    from torch import IntTensor, ByteTensor, CharTensor, FloatTensor


class TsetlinMachine2:
    """The Tsetlin Machine.

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
        self.clause_shape = (polarities, class_count,
                             self.clauses_per_class // polarities, 1)
        action_shape = (polarities, class_count,
                             self.clauses_per_class // polarities, feature_count)
        self.automata = torch.IntTensor(*action_shape).random_(states, states + 2)
        self.inv_automata = torch.IntTensor(*action_shape).random_(states, states + 2)
        self.action = torch.IntTensor(*action_shape)
        self.inv_action = torch.IntTensor(*action_shape)
        self.update_action()

        if use_cuda:
            self.cuda()
        assert isinstance(self.automata, IntTensor), type(self.automata)
        assert isinstance(self.inv_automata, IntTensor)

    def cuda(self):
        if torch.cuda.is_available():
            self.automata = self.automata.cuda()
            self.inv_automata = self.inv_automata.cuda()
            self.action = self.action.cuda()
            self.inv_action = self.inv_action.cuda()

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
                shape: (polarities, class_count, self.clauses_per_class // polarities, 1)

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

        used_row_sums = used_bits.int().sum(1)
        masked_input_row_sums = masked_input.int().sum(1)

        conjunction = used_row_sums.eq(masked_input_row_sums)
        assert type(conjunction) == ByteTensor, str(type(conjunction))

        # Repeat the above computations for the inverting automata.
        inv_input = ~input
        inv_used_bits = self.inv_action.view(-1, self.feature_count)
        inv_masked_input = inv_used_bits & inv_input.expand_as(inv_used_bits)

        inv_used_row_sums = inv_used_bits.int().sum(1)
        inv_masked_input_row_sums = inv_masked_input.int().sum(1)
        inv_conjunction = inv_used_row_sums.eq(inv_masked_input_row_sums)

        # The final output of each clause is the conjunction of:
        #   (1) conjunction of used, non-inverting inputs
        #   (2) conjunction of used, inverted inputs
        clause_result = conjunction & inv_conjunction
        assert isinstance(clause_result, ByteTensor), str(type(clause_result))
        return clause_result.view(*self.clause_shape)

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
        assert clause_outputs.shape == self.clause_shape

        ##### (polarity, class, clause_in_half_class, feature)

        # We split the clauses into positive polarity and negative polarity,
        # then compute the polarity-weighted votes.
        positive = clause_outputs[0].int()  # shape(classes, clauses_per_class // 2, 1)
        negative = clause_outputs[1].int()
        votes = (positive - negative).view(self.class_count, -1)

        # The votes are spread evenly across the classes.
        class_votes = votes.sum(dim=1)

        ########################################## Do we need this clamp ?????????????
        class_votes = class_votes.clamp(-self.threshold, self.threshold)
        assert class_votes.shape == (self.class_count, )
        return class_votes

    def predict(self, input: ByteTensor) -> IntTensor:
        """Forward inference of input.

        Args:
            input: 1D boolean input.

        Returns:
            The index of the class of the input (scalar held in tensor).
        """
        assert isinstance(input, ByteTensor)
        assert input.shape == (self.feature_count, )

        clause_outputs = self.evaluate_clauses(input)
        assert clause_outputs.shape == self.clause_shape

        class_votes = self.sum_up_class_votes(clause_outputs)
        assert class_votes.shape == (self.class_count, )

        value, index = class_votes.max(0)
        return index

    def evaluate(self, inputs: np.ndarray, targets: np.ndarray, notused) -> float:
        """Evaluate the machine on a dataset.

        Args:
            inputs: 2D array of inputs, each row is one (boolean) input vector.
            targets: 1D array of class indices, one for each input.

        Returns:
            Classification accuracy of the machine.
        """
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
            prediction = self.predict(input)
            if prediction[0] != targets[i]:
                errors += 1
        accuracy = (examples - errors) / examples
        return accuracy

    def _low_probability(self) -> ByteTensor:
        """Compute an array of low probabilities.

        Each element in the array is 1 with probability (1 / s).

        Returns:
            boolean array of shape [rows][columns]
            shape: action.shape
        """
        action_shape = self.action.shape
        return FloatTensor(*action_shape).uniform_() <= 1.0 / self.s

    def _high_probability(self) -> ByteTensor:
        """Compute an array of high probabilities.

        Each element in the array is 1 with probability (s-1 / s).

        Returns:
            boolean array of shape [rows][columns]
            shape: action_shape
        """
        action_shape = self.action.shape
        return FloatTensor(*action_shape).uniform_() <= (self.s - 1.0) / self.s

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
        assert clause_outputs.shape == self.clause_shape

        ###########################
        ### Sum up Clause Votes ###
        ###########################
        class_sum = self.sum_up_class_votes(clause_outputs)
        assert class_sum.shape == (self.class_count, )

        # Automata and action tensors are 2D, indexed by:
        #    polarity   (0 => positive, 1 => negative)
        #    class index
        #    class_clauses per polarity
        #    input feature index
        #
        #-----------------------------------------------------------------------------------
        #####################################
        ### Calculate Feedback to Clauses ###
        #####################################

        exper = True
        if exper:
            pos_feedback = ByteTensor(*self.clause_shape).zero_()
            neg_feedback = ByteTensor(*self.clause_shape).zero_()


        # Initialize feedback to clauses
        feedback_to_clauses = IntTensor(*self.clause_shape).zero_()

        # Process target
        feedback_rand = FloatTensor(2, self.clauses_per_class // 2, 1).uniform_()
        feedback_threshold = (feedback_rand <= (
                    1.0 / (self.threshold * 2)) *  (self.threshold - class_sum[target_class]))
        #feedback_to_clauses[0, target_class] += feedback_threshold[0].int()
        #feedback_to_clauses[1, target_class] -= feedback_threshold[1].int()

        if exper:
            pos_feedback[0, target_class] = feedback_threshold[0]
            neg_feedback[1, target_class] = feedback_threshold[1]

        # Process negative target
        feedback_rand = FloatTensor(2, self.clauses_per_class // 2, 1).uniform_()
        feedback_threshold = feedback_rand <= (
                    1.0 / (self.threshold * 2)) * \
                             (self.threshold + class_sum[
                                 anti_target_class])
        #feedback_to_clauses[0, anti_target_class] -= feedback_threshold[0].int()
        #feedback_to_clauses[1, anti_target_class] += feedback_threshold[1].int()

        if exper:
            neg_feedback[0, anti_target_class] = feedback_threshold[0]
            pos_feedback[1, anti_target_class] = feedback_threshold[1]



        #################################
        ### Train Individual Automata ###
        #################################

        low_prob = self._low_probability()
        high_prob = self._high_probability()

        if exper:
            pos_feedback = pos_feedback.expand_as(low_prob)
            neg_feedback = neg_feedback.expand_as(low_prob)

        # PyTorch does not (yet) properly implement NumPy style
        # broadcasting, so we fake it using the 'expand_as' method, which
        # essentially is broadcasting done by hand.
        clause_matrix = clause_outputs.expand_as(low_prob)
        inv_clause_matrix = clause_matrix ^ 1
        #feedback_matrix = feedback_to_clauses#
        # pos_feedback_matrix = (feedback_matrix > 0).expand_as(low_prob)
        #neg_feedback_matrix = (feedback_matrix < 0).expand_as(low_prob)

        #if exper:
        #    assert pos_feedback.equal(pos_feedback_matrix)
        #    assert neg_feedback.equal(neg_feedback_matrix)

        # Vectorization -- this is essentially unreadable. It replaces
        # the commented out code just below it
        X = input.expand_as(low_prob)
        inv_X = (input ^ 1).expand_as(low_prob)
        neg_low_delta = inv_clause_matrix & low_prob
        pos_delta = clause_matrix & X & high_prob
        neg_delta = clause_matrix & inv_X & low_prob
        pos_delta_inv = clause_matrix & inv_X & high_prob
        neg_delta_inv = clause_matrix & X & low_prob

        ########### No low_prob or high_prob after here

        # type 1 feedback
        self.automata += (pos_feedback & pos_delta).int()
        self.automata -= ((pos_feedback & neg_delta) | (pos_feedback & neg_low_delta)).int()

        self.inv_automata += (pos_feedback & pos_delta_inv).int()
        self.inv_automata -= ((pos_feedback & neg_delta_inv) | (pos_feedback & neg_low_delta)).int()

        # type 2 feedback
        self.automata += (neg_feedback & (clause_matrix & inv_X & ((self.action ^ 1)))).int()
        self.inv_automata += (neg_feedback & clause_matrix & X & ((self.inv_action ^ 1))).int()

        # Keep automata in bounds [0, 2 * states]
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
        assert isinstance(X, ByteTensor)
        assert isinstance(y, IntTensor)

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

'''

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

'''