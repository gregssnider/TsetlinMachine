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
    ('class_count', int32),
    ('clauses_count', int32),
    ('clauses_per_class', int32),
    ('feature_count', int32),
    ('s', float64),
    ('state_count', int32),
    ('automata', int32[:, :]),  # indices: [clause, feature]
    ('inverting_automata', int32[:, :]),  # indices: [clause, feature]
    ('action', int8[:, :]),  # indices: [clause, feature]
    ('inverting_action', int8[:, :]),  # indices: [clause, feature]
    ('clause_sign', int32[:, :]),  # indices: [class, clause]
    ('clause_output', int8[::1]),
    ('class_sum', int32[:]),
    ('feedback_to_clauses', int32[::1]),
    ('threshold', int32),

]

X_shape = 123


@jitclass(spec)
class MultiClassTsetlinMachine:

    def rand(self):
        return random.random()
        # return random.randint(0, RAND_MAX - 1)

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

        # The state of each Tsetlin Automaton is stored here. The automata are randomly initialized to either 'state_count' or 'state_count' + 1.
        self.automata = np.random.choice(
            np.array([state_count, state_count + 1]),
            size=(clauses_count, feature_count)).astype(np.int32)
        self.inverting_automata = np.random.choice(
            np.array([state_count, state_count + 1]),
            size=(clauses_count, feature_count)).astype(np.int32)
        self.action = np.zeros((clauses_count, feature_count),
                               dtype=np.int8)
        self.inverting_action = np.zeros((clauses_count, feature_count),
                                   dtype=np.int8)

        # Data structures for keeping track of which clause refers to which class, and the sign of the clause
        clause_count = np.zeros((class_count,), dtype=np.int32)
        self.clause_sign = np.zeros((class_count, clauses_count),
                                    dtype=np.int32)

        # Data structures for intermediate calculations (clause output, summation of votes, and feedback to clauses)
        self.clause_output = np.zeros(shape=(clauses_count,), dtype=np.int8)
        self.class_sum = np.zeros(shape=(class_count,), dtype=np.int32)
        self.feedback_to_clauses = np.zeros(shape=(clauses_count),
                                            dtype=np.int32)

        # Set up the Tsetlin Machine structure
        for i in range(class_count):
            clauses_per_class = clauses_count // class_count
            for j in range(clauses_per_class):
                # To allow for better vectorization, we move negative polarity
                # clauses to the second half of the subarray for the class
                if j < clauses_per_class // 2:
                    self.clause_sign[i, clause_count[i]] = 1
                else:
                    self.clause_sign[i, clause_count[i]] = -1

                clause_count[i] += 1
        self.update_action()

    def get_clause_index(self, class_index, clause_index):
        return class_index * self.clauses_per_class + clause_index

    def update_action(self):
        self.action = (self.automata > self.state_count)
        self.inverting_action = (self.inverting_automata > self.state_count)

    # Calculate the output of each clause using the actions of each Tsetline Automaton.
    # Output is stored an internal output array.
    def calculate_clause_output(self, X):
        for j in range(self.clauses_count):
            self.clause_output[j] = 1
            for k in range(self.feature_count):
                if (self.action[j, k] == 1 and X[k] == 0) or \
                        (self.inverting_action[j, k] == 1 and X[k] == 1):
                    self.clause_output[j] = 0

    # Sum up the votes for each class (this is the multiclass version of the Tsetlin Machine)
    def sum_up_class_votes(self):
        for target_class in range(self.class_count):
            self.class_sum[target_class] = 0

            for j in range(self.clauses_per_class):
                global_clause_index = self.get_clause_index(target_class, j)
                # global_clause_index = self.global_clause_index[target_class, j]
                self.class_sum[target_class] += \
                    self.clause_output[global_clause_index] * \
                    self.clause_sign[target_class, j]

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
        for target_class in range(1, self.class_count):
            if max_class_sum < self.class_sum[target_class]:
                max_class_sum = self.class_sum[target_class]
                max_class = target_class

        return max_class

    ############################################
    ### Evaluate the Trained Tsetlin Machine ###
    ############################################

    def evaluate(self, X, y, number_of_examples):
        feature_count = self.automata.shape[1]
        Xi = np.zeros((feature_count,)).astype(np.int32)

        errors = 0
        for l in range(number_of_examples):
            ###############################
            ### Calculate Clause Output ###
            ###############################

            for j in range(self.feature_count):
                Xi[j] = X[l, j]
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
            for target_class in range(1, self.class_count):
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

    def low_probability(self):
        """Compute an array of low probabilities.

        Returns:
            boolean array of shape [clauses, features]
        """
        return (np.random.random(
            (self.clauses_count, self.feature_count))
                <= 1.0 / self.s).astype(np.int8)

    def high_probability(self):
        """Compute an array of high probabilities.

        Returns:
            boolean array of shape [clauses, features]
        """
        return (np.random.random(
            (self.clauses_count, self.feature_count))
                <= (self.s - 1.0) / self.s).astype(np.int8)

    def clause_index(self, target_class: int, clause: int, pos_polarity: bool):
        index = target_class * (self.clauses_per_class // 2) + clause
        if not pos_polarity:
            index += self.clauses_count // 2

    def update(self, X, target_class):

        # Randomly pick one of the other classes, for pairwise learning of class output
        negative_target_class = target_class
        while negative_target_class == target_class:
            negative_target_class = random.randint(0,
                                                   self.class_count - 1)

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
        self.feedback_to_clauses = np.zeros_like(self.feedback_to_clauses)

        # Process target
        half = self.clauses_per_class // 2
        feedback_threshold = np.random.random((self.clauses_per_class,))
        feedback_threshold = feedback_threshold <= (
                    1.0 / (self.threshold * 2)) * \
                             (self.threshold - self.class_sum[target_class])
        start = self.get_clause_index(target_class, 0)
        mid = start + self.clauses_per_class // 2
        end = start + self.clauses_per_class
        self.feedback_to_clauses[start: mid] += feedback_threshold[:half]
        self.feedback_to_clauses[mid: end] -= feedback_threshold[half:]

        # Process negative target
        half = self.clauses_per_class // 2
        feedback_threshold = np.random.random((self.clauses_per_class,))
        feedback_threshold = feedback_threshold <= (
                    1.0 / (self.threshold * 2)) * \
                             (self.threshold + self.class_sum[
                                 negative_target_class])
        start = self.get_clause_index(negative_target_class, 0)
        mid = start + self.clauses_per_class // 2
        end = start + self.clauses_per_class
        self.feedback_to_clauses[start: mid] -= feedback_threshold[:half]
        self.feedback_to_clauses[mid: end] += feedback_threshold[half:]

        #################################
        ### Train Individual Automata ###
        #################################

        low_prob = self.low_probability()  # shape (clauses, features)
        high_prob = self.high_probability()  # shape (clauses, features)

        # The reshape trick allows us to multiply the rows of a 2D matrix,
        # with the rows of the 1D clause_output.
        clause_matrix = self.clause_output.reshape(-1, 1)
        inv_clause_matrix = clause_matrix ^ 1
        feedback_matrix = self.feedback_to_clauses.reshape(-1, 1)
        pos_feedback_matrix = (feedback_matrix > 0)
        neg_feedback_matrix = (feedback_matrix < 0)

        assert low_prob.shape == (
        self.clauses_count, self.feature_count)
        assert clause_matrix.shape == (self.clauses_count, 1)

        # Vectorization -- this is essentially unreadable. It replaces
        # the commented out code just below it
        low_delta = inv_clause_matrix * (-low_prob)
        delta = clause_matrix * (X * high_prob - (1 - X) * low_prob)
        delta_neg = clause_matrix * (-X * low_prob + (1 - X) * high_prob)

        not_action_include = (self.automata <= self.state_count)
        not_action_include_negated = (
                self.inverting_automata <= self.state_count)

        self.automata += pos_feedback_matrix * (low_delta + delta) + \
                         neg_feedback_matrix * (clause_matrix * (1 - X) * (
            not_action_include))

        self.inverting_automata += pos_feedback_matrix * (low_delta + delta_neg) + \
                             neg_feedback_matrix * clause_matrix * X * (
                                 not_action_include_negated)

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
                self.inverting_automata[j] += low_delta + delta_neg

            elif self.feedback_to_clauses[j] < 0:
                #####################################################
                ### Type II Feedback (Combats False Positives) ###
                #####################################################

                # First do this by rows (clauses)
                action_include = (self.automata[j] > self.state_count).astype(np.int32)
                action_include_negated = (self.inverting_automata[j] > self.state_count).astype(np.int32)
                self.automata[j] += clause_matrix[j] * (1 - X) * (1 - action_include)
                self.inverting_automata[j] += clause_matrix[j] * X * (1 - action_include_negated)
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
        np.clip(self.inverting_automata, smallest, biggest, self.inverting_automata)
        '''
        for j in range(self.clauses_count):
            for k in range(self.feature_count):
                self.automata[j, k] = max(
                    min(self.automata[j, k], self.state_count * 2), 1)
                self.inverting_automata[j, k] = max(
                    min(self.inverting_automata[j, k], self.state_count * 2), 1)

    ##############################################
    ### Batch Mode Training of Tsetlin Machine ###
    ##############################################

    def fit(self, X, y, number_of_examples, epochs=100):
        feature_count = self.automata.shape[1]
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

