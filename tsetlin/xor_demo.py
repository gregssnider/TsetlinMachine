import numpy as np
import torch
import time
import tsetlin
from tsetlin.machine import TsetlinMachine2

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
    training_data = np.loadtxt("../data/noisy_xor/NoisyXORTrainingData.txt")
    test_data = np.loadtxt("../data/noisy_xor/NoisyXORTestData.txt")
    X_training = torch.from_numpy(training_data[:, 0:12].astype(np.uint8))
    y_training = torch.from_numpy(training_data[:, 12].astype(np.int32))
    X_test = torch.from_numpy(test_data[:, 0:12].astype(np.uint8))
    y_test = torch.from_numpy(test_data[:, 12].astype(np.int32))

    if tsetlin.machine.use_cuda:
        X_training = X_training.cuda()
        y_training = y_training.cuda()
        X_test = X_test.cuda()
        y_test = y_test.cuda()

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
