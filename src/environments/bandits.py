"""
All sorts of bandit problem environments go here.
"""
import numpy as np
from Ranger import Range

from ai.environments import MDP, Domain


class Bandit(MDP):
    """
    The simplest multi-armed bandit problem.
    """

    def __init__(self, n_arms=2, distributions=None):
        super().__init__(states=None, actions=Domain(domain=Range.closed(1, int(n_arms))))
        self.n_arms = int(n_arms)
        if distributions is not None:
            assert len(distributions) == self.n_arms
        self._distributions = None
        self._probabilities = None
        self.distributions = distributions

    @property
    def distributions(self):
        return self._distributions

    @distributions.setter
    def distributions(self, value):
        import collections
        from functools import partial
        value = np.random.uniform(0, 1, size=self.n_arms) if value is None else value
        if not isinstance(value, collections.Iterable):
            raise ValueError
        self._distributions = []
        self._probabilities = []
        for element in value:
            if isinstance(element, float):
                self._distributions.append(partial(np.random.choice,
                                                   a=[0, 1], size=1, replace=False, p=[element, 1 - element]))
                self._probabilities.append(element)
            elif callable(element):
                self._distributions.append(element)
                self._probabilities.append(None)
            else:
                raise ValueError

    def reward(self, action):
        if not self.actions.contains(action):
            raise ValueError
        return self.distributions[action[0]-1]()[0]

    def transition(self, action):
        """
        This problem has no concept of state. Just calculate reward and return (None, reward)
        :param action: 
        :return: (None, reward)
        """
        return self.reward(action)


if __name__ == '__main__':
    bandit = Bandit()
    rewards = np.asarray([bandit.reward(bandit.actions.sample()) for _ in range(10)])
    print(rewards.sum())
