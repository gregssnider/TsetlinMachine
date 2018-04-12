import numpy as np
import random
import torch
from torch import IntTensor, ByteTensor
from random import randint
import time
from numba import jitclass
from numba import int32, float32

RAND_MAX = 1024 * 1024
def rand():
    return randint(0, RAND_MAX - 1)


########################################
### The Multiclass Tsetlin Machine #####
########################################

class MultiClassTsetlinMachine:

    # Initialization of the Tsetlin Machine
    def __init__(self, number_of_classes, number_of_clauses, number_of_features, number_of_states, s, threshold):
        self.number_of_classes = number_of_classes
        self.number_of_clauses = number_of_clauses
        self.number_of_features = number_of_features
        self.number_of_states = number_of_states
        self.s = s
        self.threshold = threshold

        # The state of each Tsetlin Automaton is stored here. The automata are
        # randomly initialized to either 'number_of_states' or 'number_of_states' + 1.
        self.ta_state = np.random.choice([self.number_of_states, self.number_of_states+1],
            size=(self.number_of_clauses, self.number_of_features, 2)).astype(dtype=np.int32)

        # Data structures for keeping track of which clause refers to which class,
        # and the sign of the clause
        self.clause_count = np.zeros((self.number_of_classes,), dtype=np.int32)
        self.clause_sign = np.zeros((self.number_of_classes,
            self.number_of_clauses), dtype=np.int32)
        self.global_clause_index = np.zeros((self.number_of_classes,
            self.number_of_clauses), dtype=np.int32)

        # Data structures for intermediate calculations (clause output,
        # summation of votes, and feedback to clauses)
        self.clause_output = np.zeros(shape=(self.number_of_clauses,), dtype=np.int32)
        self.class_sum = np.zeros(shape=(self.number_of_classes,), dtype=np.int32)
        self.feedback_to_clauses = np.zeros(shape=(self.number_of_clauses), dtype=np.int32)

        # Random feature stuff
        self.random_values = np.zeros(shape=(self.number_of_features, ), dtype=np.float32)

        # Set up the Tsetlin Machine structure
        for i in range(self.number_of_classes):
            #print('class', i)
            for j in range(self.number_of_clauses // self.number_of_classes):
                #print('    clause', j, end=' ', flush=True)

                # clause_sign[..., 0] holds the global index for the clause,
                # it has nothing to do with the sign. Ugh.
                #
                # clause_sign[..., 1] holds either 1 or -1 (alternating)
                #
                self.global_clause_index[i,self.clause_count[i]] = \
                    i*(self.number_of_clauses/self.number_of_classes) + j
                if j % 2 == 0:
                    self.clause_sign[i, self.clause_count[i]] = 1
                else:
                    self.clause_sign[i, self.clause_count[i]] = -1
                #print('clause_sign[..., 0]:', self.clause_sign[i, self.clause_count[i], 0],
                #      'clause_sign[..., 1]:', self.clause_sign[i, self.clause_count[i], 1])
                self.clause_count[i] += 1

    # Fill self.random_values with random numbers in the range [0, 1)
    def get_random_values(self):
        for i in range(self.number_of_features):
            self.random_values[i] = 1.0 * rand() / RAND_MAX


    # Calculate the output of each clause using the actions of each
    # Tsetline Automaton.
    # Output is stored an internal output array.
    def calculate_clause_output(self, X):
        for j in range(self.number_of_clauses):
            self.clause_output[j] = 1
            for k in range(self.number_of_features):
                action_include = self.action(self.ta_state[j,k,0])
                action_include_negated = self.action(self.ta_state[j,k,1])

                if (action_include == 1 and X[k] == 0) or \
                   (action_include_negated == 1 and X[k] == 1):
                    self.clause_output[j] = 0
                    break

    # Sum up the votes for each class (this is the multiclass version of the
    # Tsetlin Machine)
    def sum_up_class_votes(self):

        for target_class in range(self.number_of_classes):
            self.class_sum[target_class] = 0

            for j in range(self.clause_count[target_class]):
                self.class_sum[target_class] += \
                    self.clause_output[self.global_clause_index[target_class,j]] * \
                        self.clause_sign[target_class,j]

            if self.class_sum[target_class] > self.threshold:
                self.class_sum[target_class] = self.threshold
            elif self.class_sum[target_class] < -self.threshold:
                self.class_sum[target_class] = -self.threshold

    ########################################
    ### Predict Target Class for Input X ###
    ########################################

    def predict(self, X):
        ###############################
        ### Calculate Clause Output ###
        ###############################

        self.calculate_clause_output(X)

        ###########################
        ### Sum up Clause Votes ###
        ###########################

        self.sum_up_class_votes()

        ##########################################
        ### Identify Class with Largest Output ###
        ##########################################

        max_class_sum = self.class_sum[0]
        max_class = 0
        for target_class in range(1, self.number_of_classes):
            if max_class_sum < self.class_sum[target_class]:
                max_class_sum = self.class_sum[target_class]
                max_class = target_class

        return max_class

    # Translates automata state to action
    def action(self, state):
        if state <= self.number_of_states:
            return 0
        else:
            return 1

    # Get the state of a specific automaton, indexed by clause, feature, and
    # automaton type (include/include negated).
    def get_state(self, clause: int, feature: int, automaton_type: int):
        return self.ta_state[clause,feature,automaton_type]

    ############################################
    ### Evaluate the Trained Tsetlin Machine ###
    ############################################

    def evaluate(self, X, y, number_of_examples):

        Xi = np.zeros((self.number_of_features,), dtype=np.int32)

        errors = 0
        for l in range(number_of_examples):
            ###############################
            ### Calculate Clause Output ###
            ###############################

            for j in range(self.number_of_features):
                    Xi[j] = X[l,j]
            self.calculate_clause_output(Xi)

            ###########################
            ### Sum up Clause Votes ###
            ###########################

            self.sum_up_class_votes()

            ##########################################
            ### Identify Class with Largest Output ###
            ##########################################

            max_class_sum = self.class_sum[0]
            max_class = 0
            for target_class in range(1, self.number_of_classes):
                if max_class_sum < self.class_sum[target_class]:
                    max_class_sum = self.class_sum[target_class]
                    max_class = target_class

            if max_class != y[l]:
                errors += 1

        return 1.0 - 1.0 * errors / number_of_examples

    ##########################################
    ### Online Training of Tsetlin Machine ###
    ##########################################

    # The Tsetlin Machine can be trained incrementally, one training example at a time.
    # Use this method directly for online and incremental training.

    def update(self, X, target_class):

        # Randomly pick one of the other classes, for pairwise learning of class output
        negative_target_class = int(self.number_of_classes * 1.0*rand()/RAND_MAX)
        while negative_target_class == target_class:
            negative_target_class = int(self.number_of_classes * 1.0*rand()/RAND_MAX)

        ###############################
        ### Calculate Clause Output ###
        ###############################

        self.calculate_clause_output(X)

        ###########################
        ### Sum up Clause Votes ###
        ###########################

        self.sum_up_class_votes()

        #####################################
        ### Calculate Feedback to Clauses ###
        #####################################

        # Initialize feedback to clauses
        for j in range(self.number_of_clauses):
            self.feedback_to_clauses[j] = 0

        # Calculate feedback to clauses

        # Feedback to target class
        for j in range(self.clause_count[target_class]):
            if 1.0*rand()/RAND_MAX > (1.0/(self.threshold*2))*(self.threshold -
                                      self.class_sum[target_class]):
                continue

            global_clause_index = self.global_clause_index[target_class, j]
            self.feedback_to_clauses[global_clause_index] += \
                self.clause_sign[target_class, j]

        # Feedback to negative target class
        for j in range(self.clause_count[negative_target_class]):
            if 1.0*rand()/RAND_MAX > (1.0/(self.threshold*2))*(self.threshold +
                                      self.class_sum[negative_target_class]):
                continue

            global_clause_index = self.global_clause_index[negative_target_class, j]
            self.feedback_to_clauses[global_clause_index] -= \
                self.clause_sign[negative_target_class, j]


        #################################
        ### Train Individual Automata ###
        #################################

        for j in range(self.number_of_clauses):
            clause_output = self.clause_output[j]
            feedback = self.feedback_to_clauses[j]
            if feedback > 0:
                ####################################################
                ### Type I Feedback (Combats False Negatives) ###
                ####################################################
                if clause_output == 0:
                    self.get_random_values()
                    for k in range(self.number_of_features):
                        if self.random_values[k] <= 1.0/self.s:
                            self.ta_state[j,k,0] -= 1

                    self.get_random_values()
                    for k in range(self.number_of_features):
                        if self.random_values[k] <= 1.0/self.s:
                            self.ta_state[j,k,1] -= 1

                elif clause_output == 1:
                    self.get_random_values()
                    for k in range(self.number_of_features):
                        if self.random_values[k] <= 1.0 * (self.s-1)/self.s:
                            self.ta_state[j,k, 1-X[k]] += 1

                    self.get_random_values()
                    for k in range(self.number_of_features):
                        if self.random_values[k] <= 1.0/self.s:
                            self.ta_state[j,k,X[k]] -= 1

            elif feedback < 0:
                #####################################################
                ### Type II Feedback (Combats False Positives) ###
                #####################################################
                if clause_output == 1:
                    for k in range(self.number_of_features):
                        action_include = self.action(self.ta_state[j,k,0])
                        action_include_negated = self.action(self.ta_state[j,k,1])

                        if X[k] == 0:
                            if action_include == 0:
                                self.ta_state[j,k,0] += 1
                        elif X[k] == 1:
                            if action_include_negated == 0:
                                self.ta_state[j,k,1] += 1
            else:
                pass  # print('zero feedback')

            # Clamping automata to the range [1, 2 * number_of_states]
            for k in range(self.number_of_features):
                if self.ta_state[j, k, 0] < 1:
                    self.ta_state[j, k, 0] = 1
                if self.ta_state[j, k, 0] > self.number_of_states * 2:
                    self.ta_state[j, k, 0] = self.number_of_states * 2
                if self.ta_state[j, k, 1] < 1:
                    self.ta_state[j, k, 1] = 1
                if self.ta_state[j, k, 1] > self.number_of_states * 2:
                    self.ta_state[j, k, 1] = self.number_of_states * 2


    ##############################################
    ### Batch Mode Training of Tsetlin Machine ###
    ##############################################

    def fit(self, X, y, number_of_examples, epochs=100):
        Xi = np.zeros((self.number_of_features,), dtype=np.int32)

        random_index = np.arange(number_of_examples)
        print()
        for epoch in range(epochs):
            np.random.shuffle(random_index)

            for i in range(number_of_examples):
                example_id = random_index[i]
                target_class = y[example_id]

                for j in range(self.number_of_features):
                    Xi[j] = X[example_id,j]
                self.update(Xi, target_class)
            print('\repoch', epoch, end='', flush=True)
        print()
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
    training_data = np.loadtxt("NoisyXORTrainingData.txt").astype(dtype=np.int32)
    test_data = np.loadtxt("NoisyXORTestData.txt").astype(dtype=np.int32)

    X_training = training_data[:, 0:12]  # Input features
    y_training = training_data[:, 12]  # Target value
    X_test = test_data[:, 0:12]  # Input features
    y_test = test_data[:, 12]  # Target value

    print("Noisy XOR")
    start_time = time.time()
    tsetlin_machine = MultiClassTsetlinMachine(
        number_of_classes, number_of_clauses, number_of_features, states, s, T)
    tsetlin_machine.fit(X_training, y_training, y_training.shape[0],
                        epochs=epochs)
    elapsed_time = time.time() - start_time

    print("Accuracy on test data (no noise):",
          tsetlin_machine.evaluate(X_test, y_test, y_test.shape[0]),
          'elapsed time:', elapsed_time)


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

