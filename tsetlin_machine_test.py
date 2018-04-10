import numpy as np
import torch
from torch import ByteTensor, IntTensor
from tsetlin_machine import TsetlinMachine


def test_constructor():
    machine = TsetlinMachine(class_count=10, clause_count=300,
                             feature_count=28 * 28, state_count=100,
                             s=3.9, threshold=15)
    automata_shape = (machine.clause_count, machine.feature_count)
    assert machine.automata.shape == automata_shape
    assert machine.inverting_automata.shape == automata_shape


def calculate_clause_output(machine: TsetlinMachine, input: ByteTensor):
    """Independent and slow implementation of the evaluate_clauses() method."""
    action = machine.action.numpy()
    inverting_action = machine.inverting_action.numpy()
    input = input.numpy()

    # Manually calculate the output of the clauses. This is partly copied from
    # the author's cython implementation.
    clauses = action.shape[0]
    output = [0] * clauses
    for row in range(clauses):
        output[row] = 1
        for col in range(input.shape[0]):
            action_include = action[row, col]
            action_include_neg = inverting_action[row, col]
            if (action_include == 1 and input[col] == 0) or \
                    (action_include_neg == 1 and input[col] == 1):
                output[row] = 0
                break
    return ByteTensor(output)


def random_input(in_features: int) -> ByteTensor:
    """Return a random bit tensor of length 'in_features.'"""
    state = np.random.randint(0, 2, (in_features, )).astype(np.int32)
    return torch.from_numpy(state).byte()


def test_randomly_evaluate_clauses():
    clause_count = 100
    feature_count = 50
    for _ in range(100):
        machine = TsetlinMachine(class_count=1, clause_count=clause_count,
                                 feature_count=feature_count, state_count=3,
                                 s=3.9, threshold=15)
        input = random_input(feature_count)
        output = machine.evaluate_clauses(input)
        expected_output = calculate_clause_output(machine, input)
        assert output.equal(expected_output)


def test_evaluate_clauses():
    clause_count = 4
    feature_count = 5
    machine = TsetlinMachine(class_count=1, clause_count=clause_count,
                             feature_count=feature_count, state_count=3,
                             s=3.9, threshold=15)
    # Replace the automata with crafted ones so we can check clause evaluation.
    #    0 => corresponding input bit NOT used
    #    4 => corresponding input bit IS used
    automata_shape = (clause_count, feature_count)
    machine.automata = IntTensor([
        [4, 4, 4, 0, 0],
        [0, 4, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 4, 4, 4],
    ])
    assert machine.automata.shape == automata_shape
    machine.inverting_automata = IntTensor([
        [0, 0, 0, 4, 4],
        [0, 0, 4, 0, 0],
        [0, 0, 0, 4, 4],
        [0, 0, 0, 0, 0]
    ])
    assert machine.inverting_automata.shape == automata_shape
    machine.update_action()
    inputs = [
        ByteTensor([0, 1, 0, 1, 1]),
        ByteTensor([1, 1, 1, 0, 0]),
        ByteTensor([0, 0, 0, 0, 0]),
    ]
    expected_outputs = [
        ByteTensor([0, 1, 0, 0]),
        ByteTensor([1, 0, 1, 0]),
        ByteTensor([0, 0, 1, 0]),
    ]
    for index, input in enumerate(inputs):
        output = machine.evaluate_clauses(input)
        x = calculate_clause_output(machine, input)
        assert x.equal(output)
        assert output.equal(expected_outputs[index])
