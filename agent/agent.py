import numpy as np
import random

from utils.state import State, validate_state, iterate, discretize, to_simple_tuple
from utils.actions import num_actions, action_level
from sys import maxsize


EPSILON = .1


class Agent:

    def __init__(self, name, state, params):
        validate_state(state)

        self.name = name
        self.__history = [state]
        self.__parameters = params
        self.__learning_rate = 0.4
        self.state_action_map = {}

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
        Performs one iteration of the agent's state and appends it to the history, making it the new current state.
        """

        next_state = iterate(self.__history[-1], self.__parameters)
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

    def select_action(self, state=None):
        if state is None:
            state = self.__history[-1]

        # pick random action as default
        action = random.randint(0, num_actions() - 1)

        # If not exploring, get the best action
        if random.random() > EPSILON:
            action = self.find_best_action(state)

        return action

    def find_best_action(self, state):
        """
        Given a state, will return the action with the highest Q-value in the state_action_map. If no map can be found,
        will create one with random values.
        :return: Int denoting the index in the Actions array, as found in `utils.actions`.
        """
        state = to_simple_tuple(state)
        action_map = [random.random() for _ in range(num_actions())]  # init an action distribution map

        # Try to retrieve an existing action map
        try:
            action_map = self.state_action_map[state]
        except KeyError:
            self.state_action_map[state] = action_map

        # Return the index (a.k.a. the action) of the highest stored Q-value.
        return action_map.index(max(action_map))

    def compute_reward(self):
        """
        Computes a reward valuation for a given agent's current state.
        """

        if self.at_goal():
            return maxsize - 1000 * (len(self.__history))

        return 0

    def at_goal(self):
        """
        The agent is in a goal state if an equilibrium is reached. In practice this means when 10 successive states
        are (almost) equal to the current one.
        """

        curr_state = self.__history[-1]

        for _ in range(9):
            next_state = iterate(curr_state, self.__parameters)

            if discretize(curr_state) != discretize(next_state):
                return False

            curr_state = next_state

        return True
