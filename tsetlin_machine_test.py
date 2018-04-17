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
    action = machine.action.numpy()
    inverting_action = machine.inv_action.numpy()
    input = input.numpy()

    # Manually calculate the output of the clauses. This is partly copied from
    # the author's cython implementation.
    clauses = machine.clause_count
    output = [0] * clauses
    for polarity in (0, 1):
        for class_ in range(machine.class_count):
            for clause in range(machine.clauses_per_class // 2):
                output[clause] = 0
                for feature in range(machine.feature_count):
                    action_include = action[polarity, class_, clause, feature]
                    action_include_neg = inverting_action[polarity, class_, clause, feature]
                    if (action_include == 1 and input[feature] == 0) or \
                            (action_include_neg == 1 and input[feature] == 1):
                        output[feature] = 0
                        break
    # output shape = (machine.clause_count, )
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
        output = machine.evaluate_clauses(input).view(clause_count)
        expected_output = calculate_clause_output(machine, input)
        assert output.equal(expected_output)


def test_sum_up_class_votes():
    class_count = 2
    clause_count = 16
    feature_count = 500
    machine = TsetlinMachine2(class_count=class_count, clause_count=clause_count,
                             feature_count=feature_count, states=3,
                             s=3.9, threshold=15)
    input = ByteTensor([0, 1, 0, 1, 0] * 100)
    clause_outputs = machine.evaluate_clauses(input)
    class_votes = machine.sum_up_class_votes(clause_outputs)
    print(class_votes)
