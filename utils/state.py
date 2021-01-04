from collections import namedtuple

Parameters = namedtuple("Parameters", ["a", "b", "d", "g", "r"])


class State:

    def __init__(self, s, e, i, r, n):
        """
        Simply assign values when creating a State object
        """
        self.S = s
        self.E = e
        self.I = i
        self.R = r
        self.N = n

    def __eq__(self, other):
        return self.S == other.S and self.E == other.E and self.I == other.I and self.R == other.R

    def to_list(self):
        return [
            self.S,
            self.E,
            self.I,
            self.R,
            self.N,
        ]

    def is_valid(self):
        """
        Validates the given input by checking the type and values of the input.
        :param self: input to be checked - should be of type `State`
        """

        # Check if actually a State namedtuple
        if not isinstance(self, State):
            raise TypeError("state parameter not namedtuple State")

        # Check if each value in state is valid
        for value in self.to_list():
            if value is None or value < 0:
                raise ValueError("Value(s) in state parameter invalid, should be >=0. Please check contents.")

        return True

    def make_discrete(self):
        """
        Discretizes a state by converting the state params to ints. This is done by flooring the
        compartments after multiplying them by 100 to convert them to percentages.
        :return: A discretized State object
        """

        if not self.is_valid():
            raise ValueError("Please provide a valid state for discretizing.")

        self.S = int(self.S * 100)
        self.E = int(self.E * 100)
        self.I = int(self.I * 100)
        self.R = int(self.R * 100)

    def iterate(self, params):
        """
        Performs one iteration using the state-params combination.
        :return: State namedtuple using
        """

        # Split parameters for readability
        S, E, I, R, N = self.to_list()
        a, b, g, d, r = params

        # Compute next values
        next_S = S - (r * b * S * I) + (d * R)  # Add fraction of recovered compartment.
        next_E = E + (r * b * S * I - a * E)
        next_I = I + (a * E - g * I)
        next_R = R + (g * I) - (d * R)  # Remove fraction of recovered compartment.

        # Update current values
        self.S = next_S
        self.E = next_E
        self.I = next_I
        self.R = next_R
