import numpy as np


class QTableParams:
    def __init__(self, exploration_rate=0.5, discount_factor=0.9, learning_rate=0.1):
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate


class QTable:
    """ Works for Discrete state and action spaces
    
    """

    def __init__(self, action_dim, state_dim, params=None):
        self._q_table = np.zeros(shape=(state_dim, action_dim), dtype=float)
        self.action_dim, self.state_dim = action_dim, state_dim
        self.params = QTableParams() if params is None else params
        self.discount_factor = self.params.discount_factor
        self.learning_rate = self.params.learning_rate

    def train(self, train_dict):
        state, action, new_state, reward = (train_dict[key]
                                            for key in 'state action new_state reward'.split())
        old_state_action_value = self.state_action_value(state, action)
        max_future_reward = self.max_expected_reward_for_state(new_state)
        # Move the q_value for the state and action in the direction of the expected value.
        self._q_table[state, action] += self.learning_rate * (reward +
                                                              self.discount_factor * max_future_reward -
                                                              old_state_action_value)

    def act(self, state, stochastic=True):
        if stochastic:
            return np.argmax(self._q_table[state, :] + np.random.randn(1, self.action_dim))
        else:
            return self._q_table[state, :].argmax()

    def state_action_value(self, state, action):
        """Expected long term reward for an action in any given state
        
        :param state: 
        :param action: 
        :return: 
        """
        return self._q_table[state, action]

    def max_expected_reward_for_state(self, state):
        return self._q_table[state, :].max()
