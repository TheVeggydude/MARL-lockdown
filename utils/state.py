from collections import namedtuple

State = namedtuple("State", ["S", "E", "I", "R", "N"])
Parameters = namedtuple("Parameters", ["a", "b", "d", "g", "r"])


def validate_state(candidate):
    """
    Validates the given input by checking the type and values of the input.
    :param candidate: input to be checked - should be of type `State`
    """

    # Check if actually a State namedtuple
    if not isinstance(candidate, State):
        raise TypeError("state parameter not namedtuple State")

    # Check if each value in state is valid
    for value in candidate:
        if value is None or value < 0:
            raise ValueError("Value(s) in state parameter invalid, should be >=0. Please check contents.")


def discretize_state(cont_state):
    """
    Discretizes a state by converting the state to an int that reflects the unique state. This is done by flooring the
    compartments after multiplying them by 100 to convert them to percentages.
    :return: An Int denoting the discretized state - which corresponds to one of the input nodes of the NN
    """

    discr_state = State(
        int(cont_state.S * 100),
        int(cont_state.E * 100),
        int(cont_state.I * 100),
        int(cont_state.R * 100),
        cont_state.N
    )

    return discr_state


def compute_reward(state):
    """
    Computes a reward valuation for a given state.
    """
    
