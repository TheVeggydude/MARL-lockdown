from collections import namedtuple

State = namedtuple("State", ["S", "E", "I", "R", "N"])
State.__eq__ = lambda x, y: x.S == y.S and x.E == y.E and x.I == y.I and x.R == y.R

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

    return True


def discretize(cont_state):
    """
    Discretizes a state by converting the state to an int that reflects the unique state. This is done by flooring the
    compartments after multiplying them by 100 to convert them to percentages.
    :return: An Int denoting the discretized state - which corresponds to one of the input nodes of the NN
    """

    if not validate_state(cont_state):
        raise ValueError("Please provide a valid state for discretizing.")

    discr_state = State(
        int(cont_state.S * 100),
        int(cont_state.E * 100),
        int(cont_state.I * 100),
        int(cont_state.R * 100),
        cont_state.N
    )

    return discr_state


def iterate(state, params):
    """
    Performs one iteration using the state-params combination.
    :return: State namedtuple using
    """

    S, E, I, R, N = state
    a, b, g, d, r = params

    next_S = S - (r * b * S * I) + (d * R)  # Add fraction of recovered compartment.
    next_E = E + (r * b * S * I - a * E)
    next_I = I + (a * E - g * I)
    next_R = R + (g * I) - (d * R)  # Remove fraction of recovered compartment.

    return State(next_S, next_E, next_I, next_R, N)
