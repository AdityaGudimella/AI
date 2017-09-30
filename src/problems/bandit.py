import numpy as np

from ai.utils import argmax
from ai.environments import MDP


class Bandit(MDP):
    """
    Meh
    """

    def __init__(self, num_arms=1, *args, **kwargs):
        """

        :param num_arms:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.p = np.random.uniform(0, 1, size=num_arms)
        self.μ = 0
        self.total_value = 0
        self.num_runs = np.zeros(num_arms)
        self._values = []
        self._time = 0
        self.actions = np.arange(num_arms - 1)
        self._history = []
        self._optimal_history = []
        self._optimal_rewards = []

    def reward(self, action):
        pass

    def transition(self, action):
        pass

    # Metrics -> -------------------------------------------------------------------------------------------------------
    def average_reward(self):
        return np.mean(self._values)

    def average_performance(self):
        return sum(action == optimal_action
                   for action, optimal_action in zip(self._history, self._optimal_history)) / (self._time - 1)

    # Metrics <- -------------------------------------------------------------------------------------------------------

    def policy(self, strategy=None):
        if strategy is not None:
            assert callable(strategy)
            assert strategy(self._time) in self.actions
            return strategy(self._time)
        else:  # Greedy strategy: select action with highest estimated reward.
            return argmax(self.estimated_value, self.actions)

    @property
    def value(self, time=None):
        time = -1 if time is None else time
        return self._values[time]

    @value.setter
    def value(self, value):
        raise ValueError('Cannot set reward!')

    def true_value(self):
        """
        Defines the true reward of action a.
        :return: float
        """
        raise NotImplementedError

    def estimated_value(self, action, estimator=None):
        """
        Defines the agent's estimate of the reward at the time t
        :return: float
        """
        if estimator is not None:
            assert callable(estimator)
            return estimator(action)
        else:  # Average of all rewards obtained for action
            return sum(self._history[action]) / len(self._history[action])

    def evaluate(self, *args, **kwargs):
        super().evaluate(*args, **kwargs)
        # What should I do here?
        pass

    def play(self, n_episodes=1):
        # Todo: At each pull, append the reward to self._reward and the action to self._history. Update self._time
        current_value = np.random.binomial(n_episodes, self.p)
        # update properties
        self.μ = (self.num_runs * self.μ + current_value) / (self.num_runs + n_episodes)
        self.num_runs += n_episodes
        self.total_value += current_value

        return current_value
