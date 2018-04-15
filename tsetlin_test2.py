from tsetlin_machine2 import TsetlinMachine2

if __name__ == '__main__':
    # Parameters for the Tsetlin Machine
    T = 15
    s = 3.9
    number_of_clauses = 5 #20
    states = 100

    # Parameters of the pattern recognition problem
    number_of_features = 3 #12
    number_of_classes = 2

    tsetlin_machine = TsetlinMachine2(
        number_of_classes, number_of_clauses, number_of_features, states, s, T)
    print(tsetlin_machine)