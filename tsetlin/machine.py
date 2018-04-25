import numpy as np
import time
import torch
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('using GPU (CUDA)')
    from torch.cuda import IntTensor, ByteTensor, FloatTensor
else:
    print('using CPU')
    from torch import IntTensor, ByteTensor, FloatTensor


class TsetlinMachine2:
    """The Tsetlin Machine as a classifier.

    This is documented in the paper by Granmo:
       https://arxiv.org/pdf/1804.01508.pdf

    The learned state variables in this model are Tsetlin automata. An automata
    consists of an integer counter, a boolean 'action' (which tells if the
    counter is in the upper half of its counting range), and increment /
    decrement operations. The counting range is clamped to the range
    [1, 2 * states], where 'states' is a hyperparameter.

    The number of automata depends on:
        (1) The number of boolean input features, and
        (2) The number of clauses in the machine.

    Each clause requires 2 arrays of automata, each array having length equal to
    the number of inputs. One array manages the conjunction of non-inverting
    inputs, the other manages the conjunction of inverting inputs. The total
    number of automata in the machine is:

        2 * #clauses * #inputs

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

    A boolean clause tensor has the following shape:

        (polarities, class_count, clauses_per_class // polarities, 1)

    where:
        polarities = 2
        class_count = number of boolean outputs of classifier
        clauses_per_class = number of machine clauses allocated to a class.


    The integer automata tensor has the same structure as a clause tensor for
    the first three dimensions, but the last dimension is equal in size to the
    number of features (number of input bits). It thus has the shape:

        (polarities, class_count, clauses_per_class//polarities, feature_count)

    The inv_automata has the same structure as the automata tensor.

    :ivar class_count: Number of boolean outputs of classifier (classes).
    :ivar clause_count: Total number of clauses in the machine.
    :ivar clause_count: Total number of clauses in the machine.
    :ivar clauses_per_class: Number of clauses for each class, a mult. of 2.
    :ivar feature_count: Number of boolean inputs.
    :ivar state_count: Number of states in each Tsetlin automata.
    :ivar s: system parameter (?)
    :ivar threshold: system parameter (?)
    :ivar automata: 4D tensor of Tsetlin automata controlling clauses.
    :ivar inv_automata: 4D tensor of Tsetlin automata controlling clauses.
    :ivar action: 4D action tensor derived automata.
    :ivar inv_action: 4D inverting action tensor derived inv_automata.
    """
    def __init__(self, class_count: int, clause_count: int, feature_count: int,
                 states: int, s, threshold):
        if clause_count % (2 * class_count) != 0:
            raise ValueError("# clauses must be a multiple of (2 * # classes)")

        print('pytorch version', torch.__version__)
        print('cuda version', torch.version.cuda)

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
        """Move the machine to the GPU. """
        if torch.cuda.is_available():
            self.automata = self.automata.cuda()
            self.inv_automata = self.inv_automata.cuda()
            self.action = self.action.cuda()
            self.inv_action = self.inv_action.cuda()

    def update_action(self):
        """Update the actions from the automata, needed after learning. """
        self.action = self.automata > self.states
        self.inv_action = self.inv_automata > self.states

    def evaluate_clauses(self, input: ByteTensor) -> ByteTensor:
        """Evaluate all clauses in the array.

        :param input: Input vector. Shape (feature_count, )
        :return: Array of outputs for each clause. Shape: self.clause_shape
        """
        # Check that all set action bits are also set in the input.
        input = input.expand_as(self.action)
        matches, _ = torch.min((self.action & input).eq(self.action), 3)

        # Same check for inv_action and inv_input.
        inv_input = (~input).expand_as(self.action)
        inv_matches, _ = torch.min((self.inv_action & inv_input).
                                       eq(self.inv_action), 3)

        # Clause is true if both tests pass.
        clause_result = matches & inv_matches
        return clause_result.view(*self.clause_shape)

    def sum_up_class_votes(self, clause_outputs: ByteTensor) -> IntTensor:
        """Add up votes for all classes.

        :param clause_outputs: Array of boolean outputs for all clauses.
            Shape: self.clause_shape
        :return: Integer sum of votes for each class.
        """
        # We split the clauses into positive polarity and negative polarity,
        # then compute the polarity-weighted votes.
        positive = clause_outputs[0].int()
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

        :param input: Input vector, shape(feature_count, )
        :return: Predicted class of the input
        """
        clause_outputs = self.evaluate_clauses(input)
        class_votes = self.sum_up_class_votes(clause_outputs)
        _, index = class_votes.max(0)
        return index

    def evaluate(self, inputs: ByteTensor, targets: IntTensor, unused) -> float:
        """Evaluate the machine on a classification dataset.

        :param inputs: Array of inputs. Shape(rows, feature_count)
        :param targets: Array of target classes for inputs. Shape(rows, )
        :param unused: Compatibility parameter, not used.
        :return: Accuracy of machine on the dataset
        """
        assert isinstance(inputs, ByteTensor)
        assert inputs.shape[1] == self.feature_count
        assert isinstance(targets, IntTensor)
        assert targets.shape == (inputs.shape[0], )

        errors = 0
        examples = targets.shape[0]
        for i in range(examples):
            input = inputs[i]
            prediction = self.predict(input)
            if prediction[0] != targets[i].long():
                errors += 1
        accuracy = (examples - errors) / examples
        return accuracy

    def train(self, input: ByteTensor, target_class: int):
        """Train the machine with a single example.

        :param input: Input vector. Shape (feature_count, )
        :param target_class: Correct class for input.
        """
        clause_outputs = self.evaluate_clauses(input)
        class_sum = self.sum_up_class_votes(clause_outputs)

        #####################################
        ### Calculate Feedback to Clauses ###
        #####################################

        pos_feedback = ByteTensor(*self.clause_shape).zero_()
        neg_feedback = ByteTensor(*self.clause_shape).zero_()

        # Process negative targets
        threshold = (1.0 / (self.threshold * 2)) * \
                    (self.threshold + class_sum.float())
        threshold = threshold.view(1, self.class_count, 1, 1)
        threshold = threshold.expand(*self.clause_shape)
        feedback_rand =  FloatTensor(2, self.class_count,
                                     self.clauses_per_class // 2, 1).uniform_()
        feedback_threshold = feedback_rand <= threshold
        neg_feedback[0] = feedback_threshold[0]
        pos_feedback[1] = feedback_threshold[1]


        # Process target
        feedback_rand = FloatTensor(2, self.clauses_per_class // 2, 1).uniform_()
        feedback_threshold = (feedback_rand <= (
                    1.0 / (self.threshold * 2)) *  (self.threshold - class_sum[target_class].float()))

        pos_feedback[0, target_class] = feedback_threshold[0]
        neg_feedback[1, target_class] = feedback_threshold[1]
        neg_feedback[0, target_class] = 0
        pos_feedback[1, target_class] = 0


        #################################
        ### Train Individual Automata ###
        #################################

        low_prob = FloatTensor(*self.action.shape).uniform_() <= 1 / self.s
        high_prob = FloatTensor(*self.action.shape).uniform_() <= (self.s - 1) / self.s

        pos_feedback = pos_feedback.expand_as(low_prob)
        neg_feedback = neg_feedback.expand_as(low_prob)
        clauses = clause_outputs.expand_as(low_prob)
        not_clauses = clauses ^ 1
        X = input.expand_as(low_prob)

        #---------------------- Start CUDA
        if use_cuda:
            increment, decrement, inv_increment, inv_decrement = \
                learn(clauses, X, low_prob, high_prob, pos_feedback,
                      neg_feedback, self.action, self.inv_action)
        else:

            inv_X = (input ^ 1).expand_as(low_prob)
            notclause_low = not_clauses & low_prob & pos_feedback
            clause_x_high = clauses & X & high_prob & pos_feedback
            clause_notx_low = clauses & inv_X & low_prob & pos_feedback
            clause_notx_high = clauses & inv_X & high_prob & pos_feedback
            clause_x_low = clauses & X & low_prob & pos_feedback

            clause_notx_notaction = clauses & inv_X & (self.action ^ 1) & neg_feedback
            clause_x_noninvaction = clauses & X & (self.inv_action ^ 1) & neg_feedback

            # The learning algorithm will increment, decrement, or leave untouched
            # every automata. You can see the exclusiveness in the following logic.

            increment = clause_x_high | clause_notx_notaction
            decrement = notclause_low | clause_notx_low

            inv_increment = clause_x_noninvaction | clause_notx_high
            inv_decrement = clause_x_low | notclause_low

        #----------------------- End CUDA
        delta = increment.int() - decrement.int()
        inv_delta = inv_increment.int() - inv_decrement.int()
        self.automata += delta
        self.inv_automata += inv_delta

        # Keep automata in bounds [0, 2 * states]
        self.automata.clamp(1, 2 * self.states)
        self.inv_automata.clamp(1, 2 * self.states)

        self.update_action()

    def fit(self, X: ByteTensor, y: IntTensor, number_of_examples, epochs=100):
        """Train the network.

        :param X: Matrix of inputs, one input per row
        :param y: Vector of categories, one per row of inputs.
        :param number_of_examples: Rows in X and y.
        :param epochs: Number of training epochs.
        """
        random_index = np.arange(number_of_examples)
        print()
        for epoch in range(epochs):
            print('\r epoch', epoch, end='', flush=True)
            start_time = time.time()
            np.random.shuffle(random_index)
            for i in range(number_of_examples):
                example_id = random_index[i]
                target_class = y[example_id]
                Xi = X[example_id]
                self.train(Xi, target_class)
            elapsed_time = time.time() - start_time
            print(' time:', elapsed_time)
        return


# Cupy kernels
from .cuda_kernels import Stream, load_kernel, CUDA_NUM_THREADS, GET_BLOCKS

kernels = '''
extern "C"
__global__ void learn(char *increment, char *decrement, char *inv_increment,
    char *inv_decrement, char *clauses, char *X, char *low_prob, 
    char *high_prob, char *pos_feedback, char *neg_feedback, char *action,
    char *inv_action, int elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > elements)
        return;

    char notclause_low = ~clauses[i] & low_prob[i] & pos_feedback[i];
    char clause_x_high = clauses[i] & X[i] & high_prob[i] & pos_feedback[i];
    char clause_notx_low = clauses[i] & ~X[i] & low_prob[i] & pos_feedback[i];
    char clause_notx_high = clauses[i] & ~X[i] & high_prob[i] & pos_feedback[i];
    char clause_x_low = clauses[i] & X[i] & low_prob[i] & pos_feedback[i];
    
    char clause_notx_notaction = clauses[i] & ~X[i] & (action[i] ^ 1) & neg_feedback[i];
    char clause_x_noninvaction = clauses[i] & X[i] & (inv_action[i] ^ 1) & neg_feedback[i];
    
    // The learning algorithm will increment, decrement, or leave untouched
    // every automata. You can see the exclusiveness in the following logic.
    
    increment[i] = clause_x_high | clause_notx_notaction;
    decrement[i] = notclause_low | clause_notx_low;
    
    inv_increment[i] = clause_x_noninvaction | clause_notx_high;
    inv_decrement[i] = clause_x_low | notclause_low;
}
'''


def learn(clauses: ByteTensor, X: ByteTensor, low_prob: ByteTensor,
          high_prob: ByteTensor, pos_feedback: ByteTensor,
          neg_feedback: ByteTensor, action: ByteTensor, inv_action: ByteTensor)\
        -> (ByteTensor, ByteTensor, ByteTensor, ByteTensor):
    assert clauses.is_cuda

    clauses = clauses.contiguous()
    X = X.contiguous()
    low_prob = low_prob.contiguous()
    high_prob = high_prob.contiguous()
    pos_feedback = pos_feedback.contiguous()
    neg_feedback = neg_feedback.contiguous()


    with torch.cuda.device_of(clauses):
        polarities, classes, clauses_per_class, features = clauses.size()
        elements = clauses.numel()

        # Outputs
        incr = clauses.new(polarities, classes, clauses_per_class, features)
        decr = clauses.new(polarities, classes, clauses_per_class, features)
        inv_incr = clauses.new(polarities, classes, clauses_per_class, features)
        inv_decr = clauses.new(polarities, classes, clauses_per_class, features)

        func = load_kernel('learn', kernels)
        func(args=[incr.data_ptr(), decr.data_ptr(), inv_incr.data_ptr(),
                   inv_decr.data_ptr(),
                   clauses.data_ptr(), X.data_ptr(),
                   low_prob.data_ptr(), high_prob.data_ptr(),
                   pos_feedback.data_ptr(), neg_feedback.data_ptr(),
                   action.data_ptr(), inv_action.data_ptr(),
                   elements],
             block=(CUDA_NUM_THREADS, 1, 1),
             grid=(GET_BLOCKS(elements), 1, 1),
             stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return incr, decr, inv_incr, inv_decr
