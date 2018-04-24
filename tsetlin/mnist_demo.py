import pyximport; pyximport.install()
import numpy as np
import time
import torch
from torch import IntTensor, ByteTensor
from torchvision import datasets
import reference.MultiClassTsetlinMachine
from tsetlin.machine import TsetlinMachine2, use_cuda


def mnist_dataset(training: bool, use_cuda=False) -> (IntTensor, ByteTensor):
    dataset = datasets.MNIST('../data/mnist', train=training, download=True)
    rows = len(dataset)
    cols = 28 * 28
    X = np.zeros((rows, cols), dtype=np.int32)
    y = np.zeros((rows, ), dtype=np.int32)
    print(X.shape)
    for i in range(rows):
        image, target = dataset[i]
        X[i] = (np.array(image).flatten() // 128).astype(np.int32)
        y[i] = target

    X = torch.from_numpy(X.astype(np.uint8))
    y = torch.from_numpy(y.astype(np.int32))
    if use_cuda:
        X = X.cuda()
        y = y.cuda()
    return X, y


if __name__ == '__main__':

    # Parameters for the Tsetlin Machine
    T = 15
    s = 3.9
    number_of_clauses = 1600
    states = 1000

    # Parameters of the pattern recognition problem
    number_of_features = 28 * 28
    number_of_classes = 10

    # Training configuration
    epochs = 10

    # Clip training to 10000 examples
    X_training, y_training = mnist_dataset(training=True, use_cuda=use_cuda)
    X_training = X_training[:10000, :]
    y_training = y_training[:10000]

    X_test, y_test = mnist_dataset(training=False, use_cuda=use_cuda)


    # This is a multiclass variant of the Tsetlin Machine, capable of distinguishing between multiple classes
    #
    print('     new on MNIST: ', end='', flush=True)
    start_time = time.time()
    tsetlin_machine = TsetlinMachine2(number_of_classes, number_of_clauses, number_of_features, states, s, T)
    tsetlin_machine.fit(X_training, y_training, y_training.shape[0], epochs)
    elapsed_time = time.time() - start_time
    print("Accuracy:", tsetlin_machine.evaluate(X_test, y_test, y_test.shape[0]),
          'time', elapsed_time)
    '''
    print('original on MNIST: ', end='', flush=True)
    start_time = time.time()
    tsetlin_machine = MultiClassTsetlinMachine.OriginalMultiClassTsetlinMachine(number_of_classes, number_of_clauses, number_of_features, states, s, T)
    tsetlin_machine.fit(X_training, y_training, y_training.shape[0], epochs=epochs)
    elapsed_time = time.time() - start_time
    print("Accuracy:", tsetlin_machine.evaluate(X_test, y_test, y_test.shape[0]),
          'time', elapsed_time)

    print('     new on MNIST: ', end='', flush=True)
    start_time = time.time()
    tsetlin_machine = MultiClassTsetlinMachine(number_of_classes, number_of_clauses, number_of_features, states, s, T)
    tsetlin_machine.fit(X_training, y_training, y_training.shape[0], epochs)
    elapsed_time = time.time() - start_time
    print("Accuracy:", tsetlin_machine.evaluate(X_test, y_test, y_test.shape[0]),
          'time', elapsed_time)

    print('original on MNIST: ', end='', flush=True)
    start_time = time.time()
    tsetlin_machine = OriginalMultiClassTsetlinMachine.OriginalMultiClassTsetlinMachine(number_of_classes, number_of_clauses, number_of_features, states, s, T)
    tsetlin_machine.fit(X_training, y_training, y_training.shape[0], epochs=epochs)
    elapsed_time = time.time() - start_time
    print("Accuracy:", tsetlin_machine.evaluate(X_test, y_test, y_test.shape[0]),
          'time', elapsed_time)

    #print("Accuracy on training data:", tsetlin_machine.evaluate(X_training, y_training, y_training.shape[0]))
    '''

