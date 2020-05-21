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
        self.comps = np.array(compartments)  # S, E, I, R

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

        mu = self.vitals[0]
        nu = self.vitals[1]

        # Compute deltas
        deltas = np.array([
            mu*N - nu*S - (beta*S*I / N),
            (beta*S*I / N) - nu*E - sigma*E,
            sigma*E - gamma*I - nu*I,
            gamma*I - nu*R
        ])

        # Change population compartments
        self.comps = self.comps + deltas
