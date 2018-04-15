import numpy as np
import torch
from torch import ByteTensor, IntTensor
#from tsetlin_machine import TsetlinMachine#
from tsetlin_machine2 import TsetlinMachine2


def test_constructor():
    machine = TsetlinMachine2(class_count=10, clause_count=300,
                             feature_count=28 * 28, states=100,
                             s=3.9, threshold=15)
    automata_shape = (2,
                      machine.class_count,
                      machine.clause_count // machine.class_count // 2,
                      machine.feature_count)
    assert machine.automata.shape == automata_shape
    assert machine.inv_automata.shape == automata_shape


def calculate_clause_output(machine: TsetlinMachine2, input: ByteTensor):
    """Independent and slow implementation of the evaluate_clauses() method."""
    features = machine.feature_count
    action = machine.action.view(-1, features).numpy()
    inverting_action = machine.inv_action.view(-1, features).numpy()
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
        machine = TsetlinMachine2(class_count=1, clause_count=clause_count,
                                 feature_count=feature_count, states=3,
                                 s=3.9, threshold=15)
        input = random_input(feature_count)
        output = machine.evaluate_clauses(input)
        expected_output = calculate_clause_output(machine, input)
        assert output.equal(expected_output)


def test_evaluate_clauses():
    clause_count = 4
    feature_count = 5
    machine = TsetlinMachine2(class_count=1, clause_count=clause_count,
                             feature_count=feature_count, states=3,
                             s=3.9, threshold=15)
    # Replace the automata with crafted ones so we can check clause evaluation.
    #    0 => corresponding input bit NOT used
    #    4 => corresponding input bit IS used
    automata_shape = (2, 1, 2, 5)
    machine.automata = IntTensor([
        [4, 4, 4, 0, 0],
        [0, 4, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 4, 4, 4],
    ]).view(*automata_shape)
    assert machine.automata.shape == automata_shape
    machine.inverting_automata = IntTensor([
        [0, 0, 0, 4, 4],
        [0, 0, 4, 0, 0],
        [0, 0, 0, 4, 4],
        [0, 0, 0, 0, 0]
    ]).view(*automata_shape)
    assert machine.inverting_automata.shape == automata_shape
    machine.update_action()
    inputs = [
        ByteTensor([0, 1, 0, 1, 1]),
        ByteTensor([1, 1, 1, 0, 0]),
        ByteTensor([0, 0, 0, 0, 0]),
    ]
    for index, input in enumerate(inputs):
        output = machine.evaluate_clauses(input)
        x = calculate_clause_output(machine, input)
        assert x.equal(output)


def test_sum_up_class_votes():
    class_count = 2
    clause_count = 16
    feature_count = 5
    machine = TsetlinMachine2(class_count=class_count, clause_count=clause_count,
                             feature_count=feature_count, states=3,
                             s=3.9, threshold=15)
    polarities = 2
    clauses_per_class = clause_count // class_count
    clause_outputs = torch.ByteTensor([
        1, 0, 1, 1,                # pos polarity class 0
        0, 1, 0, 0,                # pos polarity class 1
        0, 0, 0, 1,                # neg polarity class 0
        1, 1, 1, 1                 # neg polarity class 1
    ])

    ##################
    x = clause_outputs.view(2, class_count, -1)
    positive = x[0].int()
    negative = x[1].int()
    votes = torch.sum(positive - negative, dim=1)
    votes = torch.clamp(votes, -15, 15)
    print('pos')
    print(positive)
    print('neg')
    print(negative)
    print('votes')
    print(votes)


    ##################

    expected_votes = torch.IntTensor(
        [2, -3]
    )
    votes = machine.sum_up_class_votes(clause_outputs)
    print(votes)
    assert votes.equal(expected_votes)
