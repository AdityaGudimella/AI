"""
Defines the base agent class with a few defaults thrown in for good measure.
"""
import enum

import numpy as np

from ai.environments import MDP
from ai.utils import argmax

__author__ = 'Aditya Gudimella'


class Policies(enum.Enum):
    Custom = 'custom'
    Greedy = 'greedy'
    EpsilonGreedy = 'epsilon-greedy'


class Agent:
    """
    Base Agent class
    
    Allowed kwargs:
    """

    def __init__(self, env: MDP, policy='greedy', **kwargs):
        """
        
        :param env: 
        :param policy: string or callable. 
        If string, should be one of ['custom', 'greedy', 'epsilon-greedy']
        If callable, should take in an object of class MDP as argument and return an action from the object's action 
        space as output.
        :param kwargs: 
        """
        # TODO: take a copy of env instead of env directly?
        self.env = env
        self.horizon_length = None  # Number of time steps over which expected reward is to be calculated
        if callable(policy):
            self._policy_type = 'custom'
            self._policy = policy
        else:
            self._policy_type = policy
        self.reward_history = []
        self.kwargs = kwargs
        pass

    @property
    def allowed_kwargs(self):
        return ['epsilon']

    @property
    def implemented_policies(self):
        return [x.name for x in Policies]

    @implemented_policies.setter
    def implemented_policies(self, value):
        raise ValueError('Cannot modify variable implemented_policies')

    def policy(self):
        """
        Gives the best action to take for the current state that env is in.
        
        The policy function is a (functional) mapping from a set of states to actions, where the action corresponding
        to a given state is the action which maximizes expected reward.
        :return: 
        """
        if Policies(self._policy_type) not in Policies:
            raise ValueError(f'The policy {self._policy_type} is not in the implemented policies: '
                             f'{str(self.implemented_policies)}')
        # Greedy approach
        if Policies(self._policy_type) is Policies.Greedy:
            # choose policy which maximizes reward
            return self._greedy_policy()
        elif Policies(self._policy_type) is Policies.EpsilonGreedy:
            assert 'epsilon' in self.kwargs, 'Provide epsilon for EpsilonGreedy policy'
            epsilon = self.kwargs['epsilon']
            assert 0. <= epsilon <= 1.
            explore = np.random.choice([True, False], 1, replace=False, p=[epsilon, 1-epsilon])
            if explore:
                return self.env.actions.sample(size=1)
            else:
                return self._greedy_policy()
            pass
        elif Policies(self._policy_type) is Policies.Custom:
            return self._policy(self.env)

    def value(self, action):
        return np.mean(self.reward_history[action])

    def _greedy_policy(self):
        """
        choose policy which maximizes reward
        :return: 
        """
        return argmax(self.env.reward, self.env.actions)
