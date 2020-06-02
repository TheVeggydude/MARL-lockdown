import numpy as np


class Agent:
    """
    Class for describing SEIR agents in a network.
    """

    def __init__(self, compartments, transitions, vitals):
        """

        :param compartments:
        :param transitions:
        :param vitals:
        """

        # Population compartments
        S_0, E_0, I_0, R_0 = compartments
        self.S, self.E, self.I, self.R = [S_0], [E_0], [I_0], [R_0]

        # Transition coefficients
        self.coefs = transitions  # Beta, Sigma, Gamma, Xi

        # Vital coefficients
        self.vitals = vitals  # Mu, Nu

    def __str__(self):
        """
        Serializes Agent instance to string.
        :return:
        """
        pass

    def get_history(self):
        pass

    def next(self):
        """
        Computes the next iteration of this agent's compartments.
        :return:
        """
        N = self.comps.sum()
        
        S = self.comps[0]
        E = self.comps[1]
        I = self.comps[2]
        R = self.comps[3]

        beta = self.coefs[0]
        sigma = self.coefs[1]
        gamma = self.coefs[2]

        # Compute deltas
        deltas = np.array([
            S - (beta*S*I / N),
            (beta*S*I / N) - sigma*E,
            sigma*E - gamma*I,
            gamma*I
        ])

        # Change population compartments
        self.comps = self.comps + deltas
