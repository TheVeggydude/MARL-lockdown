import numpy as np

from collections import namedtuple

State = namedtuple("State", ["S", "E", "I", "R"])
Parameters = namedtuple("Parameters", ["a", "b", "d", "g", "r"])


def validate_state(candidate):

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
        S, E, I, R = self.__history[-1]
        a, b, d, g, r = self.__parameters

        next_S = S - (r * b * S * I) + (d * R)  # Add fraction of recovered compartment.
        next_E = E + (r * b * S * I - a * E)
        next_I = I + (a * E - g * I)
        next_R = R + (g * I) - (d * R)  # Remove fraction of recovered compartment.
        # next_C = C + (r * 3)  # Cost!

        next_state = State(next_S, next_E, next_I, next_R)
        self.__history.append(next_state)


if __name__ == '__main__':
    a = 0.2
    b = 1.75
    g = 0.5
    d = 0.2
    r = 1

    initial_state = State(0.99, 0, 0.01, 0)
    initial_params = Parameters(a, b, d, g, r)

    test = Agent(initial_state, initial_params)
    test.iterate()
