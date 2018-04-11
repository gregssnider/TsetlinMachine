import pyximport; pyximport.install()
import numpy as np
from torchvision import datasets

import OriginalMultiClassTsetlinMachine
import MultiClassTsetlinMachine

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

if __name__ == '__main__':

    # Parameters for the Tsetlin Machine
    T = 15
    s = 3.9
    number_of_clauses = 1600
    states = 200

    # Parameters of the pattern recognition problem
    number_of_features = 28 * 28
    number_of_classes = 10

    # Training configuration
    epochs = 2


    # Loading of training and test data
    training_data = np.loadtxt("NoisyXORTrainingData.txt").astype(dtype=np.int32)
    test_data = np.loadtxt("NoisyXORTestData.txt").astype(dtype=np.int32)

    X_training, y_training = mnist_dataset(training=True)
    X_test, y_test = mnist_dataset(training=False)


    # This is a multiclass variant of the Tsetlin Machine, capable of distinguishing between multiple classes
    #
    print('original on MNIST: ', end='', flush=True)
    tsetlin_machine = OriginalMultiClassTsetlinMachine.OriginalMultiClassTsetlinMachine(number_of_classes, number_of_clauses, number_of_features, states, s, T)
    tsetlin_machine.fit(X_training, y_training, y_training.shape[0], epochs=epochs)
    print("Accuracy:", tsetlin_machine.evaluate(X_test, y_test, y_test.shape[0]))

    print('     new on MNIST: ', end='', flush=True)
    tsetlin_machine = MultiClassTsetlinMachine.MultiClassTsetlinMachine(number_of_classes, number_of_clauses, number_of_features, states, s, T)
    tsetlin_machine.fit(X_training, y_training, y_training.shape[0], epochs=epochs)
    print("Accuracy:", tsetlin_machine.evaluate(X_test, y_test, y_test.shape[0]))

    print('original on MNIST: ', end='', flush=True)
    tsetlin_machine = OriginalMultiClassTsetlinMachine.OriginalMultiClassTsetlinMachine(number_of_classes, number_of_clauses, number_of_features, states, s, T)
    tsetlin_machine.fit(X_training, y_training, y_training.shape[0], epochs=epochs)
    print("Accuracy:", tsetlin_machine.evaluate(X_test, y_test, y_test.shape[0]))

    print('     new on MNIST: ', end='', flush=True)
    tsetlin_machine = MultiClassTsetlinMachine.MultiClassTsetlinMachine(number_of_classes, number_of_clauses, number_of_features, states, s, T)
    tsetlin_machine.fit(X_training, y_training, y_training.shape[0], epochs=epochs)
    print("Accuracy:", tsetlin_machine.evaluate(X_test, y_test, y_test.shape[0]))
    #print("Accuracy on training data:", tsetlin_machine.evaluate(X_training, y_training, y_training.shape[0]))


