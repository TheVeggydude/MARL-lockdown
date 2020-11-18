import numpy as np

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


class Agent:

    def __init__(self, state, params):
        validate_state(state)

        self.__history = [state]
        self.__parameters = params

    def state(self):
        """
        Gets the current state of the agent.
        :return: State namedtuple containing S, E, I and R compartments.
        """

        return self.__history[-1]

    def set_state(self, state):
        """
        Sets the current state of the Agent to a presented tuple.
        :param state: State namedtuple.
        """

        validate_state(state)
        self.__history[-1] = state

    def history(self):
        """
        Gets the collection of historical states
        :return: Numpy 2D array of states, ordered from oldest to most recent. Each row is one state of the form [S,
        E, I, R].
        """

        return np.asarray(self.__history)

    def iterate(self):
        """
        Performs one iteration of the model update loop.
        :return:
        """

        S, E, I, R, N = self.__history[-1]
        a, b, g, d, r = self.__parameters

        next_S = S - (r * b * S * I) + (d * R)  # Add fraction of recovered compartment.
        next_E = E + (r * b * S * I - a * E)
        next_I = I + (a * E - g * I)
        next_R = R + (g * I) - (d * R)  # Remove fraction of recovered compartment.
        # next_C = C + (r * 3)  # Cost!

        next_state = State(next_S, next_E, next_I, next_R, N)
        self.__history.append(next_state)

    def emigrate(self, fraction):
        """
        Generates a State, describing a population slice. This population slice is then deducted from the Agent's
        population pool.
        :param fraction: float in range 0.0 < x < 1.0 denoting what fraction of the population is emigrating.
        """

        if not 0.0 < fraction < 1.0:
            raise ValueError("Emigration fraction not in valid range 0.0 < x < 1.0")

        # Compute emigrant slice
        S, E, I, R, N = self.__history[-1]
        n_emigrants = int(N * fraction)  # int() floors/truncates the number.

        # Update the current Agent's population count
        new_state = State(S, E, I, R, N - n_emigrants)
        self.set_state(new_state)

        return State(S, E, I, R, n_emigrants)

    def immigrate(self, immigration_slice):

        # Get immigrant data
        n_im = immigration_slice.N
        distribution_im = np.asarray(immigration_slice)[:4]

        # Get local data
        n_local = self.__history[-1].N
        distribution_local = np.asarray(self.__history[-1])[:4]

        # Compute new local distribution
        distribution_post = np.add(distribution_im * n_im, distribution_local * n_local) / (n_im + n_local)

        # Replace the current state with a state that includes the new immigration
        state_post = State(
            distribution_post[0],
            distribution_post[1],
            distribution_post[2],
            distribution_post[3],
            n_im + n_local
        )
        self.__history[-1] = state_post
