import numpy as np
import random
import torch
from torch import IntTensor, ByteTensor, CharTensor
import random
import time
from numba import jitclass, jit
from numba import int8, int32, float32, int64, float64
import sys

'''
RAND_MAX = 1024 * 1024
def rand():
    return randint(0, RAND_MAX - 1)
'''

########################################
### The Multiclass Tsetlin Machine #####
########################################

spec = [
    ('class_count', int64),
    ('clauses_count', int64),
    ('clauses_per_class', int64),
    ('feature_count', int64),
    ('state_count', int64),
    ('s', float64),
    ('threshold', int32),
    ('automata', int32[:, :, :, :]),  # indices: [clause, feature]
    ('inv_automata', int32[:, :, :, :]),  # indices: [clause, feature]
    ('action', int8[:, :, :, :]),  # indices: [clause, feature]
    ('inv_action', int8[:, :, :, :]),  # indices: [clause, feature]

]

X_shape = 123


@jitclass(spec)
class MultiClassTsetlinMachine:
    # Initialization of the Tsetlin Machine
    def __init__(self, class_count, clauses_count, feature_count,
                 state_count, s, threshold):

        self.class_count = class_count
        self.clauses_count = clauses_count
        self.clauses_per_class = clauses_count // class_count
        self.feature_count = feature_count
        self.state_count = state_count
        self.s = s
        self.threshold = threshold

        action_shape = (2, class_count, self.clauses_per_class // 2, feature_count)

        # The state of each Tsetlin Automaton is stored here. The automata are randomly initialized to either 'state_count' or 'state_count' + 1.
        self.automata = np.random.choice(
            np.array([state_count, state_count + 1]),
            size=action_shape).astype(np.int32)
        self.inv_automata = np.random.choice(
            np.array([state_count, state_count + 1]),
            size=action_shape).astype(np.int32)
        self.action = np.zeros(action_shape, dtype=np.int8)
        self.inv_action = np.zeros(action_shape, dtype=np.int8)

        self.update_action()

    def get_clause_index(self, polarity, class_index, clause_index):
        assert clause_index < self.clauses_per_class // 2
        index = class_index * self.clauses_per_class + clause_index
        if polarity == 1:
            index += self.clauses_per_class // 2
        return index

    def update_action(self):
        self.action = (self.automata > self.state_count)
        self.inv_action = (self.inv_automata > self.state_count)

    # Calculate the output of each clause using the actions of each Tsetline Automaton.
    # Output is stored an internal output array.
    def calculate_clause_output(self, X):
        clause_shape = (2, self.class_count, self.clauses_per_class // 2, 1)
        clause_output = np.zeros(shape=clause_shape, dtype=np.int8)
        for polarity in (0, 1):
            for class_ in range(self.class_count):
                for clause in range(self.clauses_per_class // 2):
                    clause_output[polarity, class_, clause] = 1
                    for f in range(self.feature_count):
                        if (self.action[polarity, class_, clause, f] == 1 and X[f] == 0) or \
                                (self.inv_action[polarity, class_, clause, f] == 1 and X[f] == 1):
                            clause_output[polarity, class_, clause, 0] = 0
        return clause_output

    # Sum up the votes for each class (this is the multiclass version of the Tsetlin Machine)
    def sum_up_class_votes(self, clause_output):
        clause_shape = (2, self.class_count, self.clauses_per_class // 2, 1)
        assert clause_output.shape == clause_shape
        class_sum = np.zeros(shape=(class_count,), dtype=np.int32)
        for target_class in range(self.class_count):
            for clause in range(self.clauses_per_class // 2):
                class_sum[target_class] += clause_output[0, target_class, clause, 0]
            for clause in range(self.clauses_per_class // 2):
                class_sum[target_class] -= clause_output[1, target_class, clause, 0]
            if class_sum[target_class] > self.threshold:
                class_sum[target_class] = self.threshold
            elif class_sum[target_class] < -self.threshold:
                class_sum[target_class] = -self.threshold
        return class_sum

    ########################################
    ### Predict Target Class for Input X ###
    ########################################

    def predict(self, X):
        ###############################
        ### Calculate Clause Output ###
        ###############################

        clause_outputs = self.calculate_clause_output(X)
        clause_shape = (2, self.class_count, self.clauses_per_class // 2, 1)
        assert clause_outputs.shape == clause_shape


        ###########################
        ### Sum up Clause Votes ###
        ###########################

        class_sum = self.sum_up_class_votes(clause_outputs)

        ##########################################
        ### Identify Class with Largest Output ###
        ##########################################

        max_class_sum = class_sum[0]
        max_class = 0
        for target_class in range(1, self.class_count):
            if max_class_sum < class_sum[target_class]:
                max_class_sum = class_sum[target_class]
                max_class = target_class

        return max_class

    ############################################
    ### Evaluate the Trained Tsetlin Machine ###
    ############################################

    def evaluate(self, X, y, number_of_examples):
        Xi = np.zeros((self.feature_count,)).astype(np.int32)

        errors = 0
        for l in range(number_of_examples):
            ###############################
            ### Calculate Clause Output ###
            ###############################

            for j in range(self.feature_count):
                Xi[j] = X[l, j]
            clause_outputs = self.calculate_clause_output(Xi)
            clause_shape = (2, self.class_count, self.clauses_per_class // 2, 1)
            assert clause_outputs.shape == clause_shape

            ###########################
            ### Sum up Clause Votes ###
            ###########################

            class_sum = self.sum_up_class_votes(clause_outputs)

            ##########################################
            ### Identify Class with Largest Output ###
            ##########################################

            max_class_sum = class_sum[0]
            max_class = 0
            for target_class in range(1, self.class_count):
                if max_class_sum < class_sum[target_class]:
                    max_class_sum = class_sum[target_class]
                    max_class = target_class

            if max_class != y[l]:
                errors += 1

        return 1.0 - 1.0 * errors / number_of_examples

    ##########################################
    ### Online Training of Tsetlin Machine ###
    ##########################################

    # The Tsetlin Machine can be trained incrementally, one training example at a time.
    # Use this method directly for online and incremental training.

    def low_probability(self):
        """Compute an array of low probabilities.

        Returns:
            boolean array of shape [clauses, features]
        """
        action_shape = self.action.shape
        return (np.random.random(action_shape)
                <= 1.0 / self.s).astype(np.int8)

    def high_probability(self):
        """Compute an array of high probabilities.

        Returns:
            boolean array of shape [clauses, features]
        """
        action_shape = self.action.shape
        return (np.random.random(action_shape)
                <= (self.s - 1.0) / self.s).astype(np.int8)

    def update(self, X, target_class):

        # Randomly pick one of the other classes, for pairwise learning of class output
        negative_target_class = target_class
        while negative_target_class == target_class:
            negative_target_class = random.randint(0,
                                                   self.class_count - 1)

        ###############################
        ### Calculate Clause Output ###
        ###############################
        clause_outputs = self.calculate_clause_output(X)
        clause_shape = (2, self.class_count, self.clauses_per_class // 2, 1)
        assert clause_outputs.shape == clause_shape

        ###########################
        ### Sum up Clause Votes ###
        ###########################

        class_sum = self.sum_up_class_votes(clause_outputs)

        #####################################
        ### Calculate Feedback to Clauses ###
        #####################################

        # Initialize feedback to clauses
        feedback_to_clauses = np.zeros(shape=clause_shape, dtype=np.int32)

        # Process target
        half = self.clauses_per_class // 2
        feedback_rand = np.random.random((2, self.clauses_per_class // 2, 1))
        feedback_threshold = feedback_rand <= (
                    1.0 / (self.threshold * 2)) *  (self.threshold - class_sum[target_class])
        feedback_to_clauses[0, target_class] += feedback_threshold[0]
        feedback_to_clauses[1, target_class] -= feedback_threshold[1]

        # Process negative target
        feedback_rand = np.random.random((2, self.clauses_per_class // 2, 1))
        feedback_threshold = feedback_rand <= (
                    1.0 / (self.threshold * 2)) * \
                             (self.threshold + class_sum[
                                 negative_target_class])
        feedback_to_clauses[0, negative_target_class] -= feedback_threshold[0]
        feedback_to_clauses[1, negative_target_class] += feedback_threshold[1]

        #################################
        ### Train Individual Automata ###
        #################################

        low_prob = self.low_probability()
        high_prob = self.high_probability()

        # The reshape trick allows us to multiply the rows of a 2D matrix,
        # with the rows of the 1D clause_output.
        clause_matrix = clause_outputs
        inv_clause_matrix = clause_matrix ^ 1
        feedback_matrix = feedback_to_clauses
        pos_feedback_matrix = (feedback_matrix > 0)
        neg_feedback_matrix = (feedback_matrix < 0)

        # Vectorization -- this is essentially unreadable. It replaces
        # the commented out code just below it
        low_delta = inv_clause_matrix * (-low_prob)
        delta = clause_matrix * (X * high_prob - (1 - X) * low_prob)
        delta_neg = clause_matrix * (-X * low_prob + (1 - X) * high_prob)

        self.automata += pos_feedback_matrix * (low_delta + delta) + \
                         neg_feedback_matrix * (clause_matrix * (1 - X) * (
            (self.action ^ 1)))

        self.inv_automata += pos_feedback_matrix * (low_delta + delta_neg) + \
                             neg_feedback_matrix * clause_matrix * X * (
                                 (self.inv_action ^ 1))

        '''
        for j in range(self.clauses_count):
            if self.feedback_to_clauses[j] > 0:
                ####################################################
                ### Type I Feedback (Combats False Negatives) ###
                ####################################################

                # First do this by rows (clauses)
                low_delta = inv_clause_matrix[j] * (-low_prob[j])
                delta = clause_matrix[j] * (X * high_prob[j] - (1-X) * low_prob[j])
                delta_neg = clause_matrix[j] * (
                            -X * low_prob[j] + (1 - X) * high_prob[j])
                self.automata[j] += low_delta + delta
                self.inv_automata[j] += low_delta + delta_neg

            elif self.feedback_to_clauses[j] < 0:
                #####################################################
                ### Type II Feedback (Combats False Positives) ###
                #####################################################

                # First do this by rows (clauses)
                action_include = (self.automata[j] > self.state_count).astype(np.int32)
                action_include_negated = (self.inv_automata[j] > self.state_count).astype(np.int32)
                self.automata[j] += clause_matrix[j] * (1 - X) * (1 - action_include)
                self.inv_automata[j] += clause_matrix[j] * X * (1 - action_include_negated)
        '''

        self.clamp_automata()
        self.update_action()

    def clamp_automata(self):
        """Clamp all automata states to the range[1, 2*state_count]."""
        '''
        # np.clip not supported by jit
        smallest = 1
        biggest = self.state_count * 2
        np.clip(self.automata, smallest, biggest, self.automata)
        np.clip(self.inv_automata, smallest, biggest, self.inv_automata)
        '''
        for polarity in (0, 1):
            for class_ in range(self.class_count):
                for clause in range(self.clauses_per_class // 2):
                    for f in range(self.feature_count):
                        if self.automata[polarity, class_, clause, f] < 1:
                            self.automata[polarity,class_, clause, f] = 1
                        if self.automata[polarity, class_, clause, f] > 2 * self.state_count:
                            self.automata[polarity,class_, clause, f] = 2 * self.state_count
                        if self.inv_automata[polarity, class_, clause, f] < 1:
                            self.inv_automata[polarity,class_, clause, f] = 1
                        if self.inv_automata[polarity, class_, clause, f] > 2 * self.state_count:
                            self.inv_automata[polarity,class_, clause, f] = 2 * self.state_count


    ##############################################
    ### Batch Mode Training of Tsetlin Machine ###
    ##############################################

    def fit(self, X, y, number_of_examples, epochs=100):
        feature_count = self.automata.shape[3]
        Xi = np.zeros((feature_count,)).astype(np.int32)

        random_index = np.arange(number_of_examples)

        for epoch in range(epochs):
            np.random.shuffle(random_index)

            for i in range(number_of_examples):
                example_id = random_index[i]
                target_class = y[example_id]

                for j in range(self.feature_count):
                    Xi[j] = X[example_id, j]
                self.update(Xi, target_class)
        return


if __name__ == '__main__':
    # Parameters for the Tsetlin Machine
    T = 15
    s = 3.9
    clauses_count = 20
    states = 100

    # Parameters of the pattern recognition problem
    feature_count = 12
    class_count = 2

    # Training configuration
    epochs = 200

    # Loading of training and test data
    training_data = np.loadtxt("NoisyXORTrainingData.txt").astype(dtype=np.int8)
    test_data = np.loadtxt("NoisyXORTestData.txt").astype(dtype=np.int8)
    X_training = training_data[:, 0:12]  # Input features
    y_training = training_data[:, 12]  # Target value
    X_test = test_data[:, 0:12]  # Input features
    y_test = test_data[:, 12]  # Target value

    print("Noisy XOR, numpy version")
    sum_accuracy = 0
    steps = 50
    for step in range(steps):
        start_time = time.time()
        tsetlin_machine = MultiClassTsetlinMachine(
            class_count, clauses_count, feature_count, states, s,
            T)
        tsetlin_machine.fit(X_training, y_training, y_training.shape[0], epochs)
        elapsed_time = time.time() - start_time
        accuracy = tsetlin_machine.evaluate(X_test, y_test, y_test.shape[0])
        print("  ", step, " Accuracy on test data (no noise):", accuracy,
              ', elapsed time:', elapsed_time)
        sum_accuracy += accuracy
    print('Avg accuracy', sum_accuracy / steps)

