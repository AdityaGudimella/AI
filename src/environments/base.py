"""
Defines base classes for all environments.
"""
from abc import ABC, abstractmethod

__author__ = 'Aditya Gudimella'


class Environment(ABC):
    def __init__(self, name):
        self.name = name

    @property
    @abstractmethod
    def states(self):
        return NotImplemented

    @property
    @abstractmethod
    def actions(self):
        return NotImplemented

    def render(self):
        """ Implements all GUI rendering here
        
        Subclass this method to implement any rendering for the environment
        
        :return: None
        """
        pass

    @abstractmethod
    def step(self):
        """
        
        :return: Following openAI's convention, should return (new_state, reward, done, info) 
        """
        return NotImplemented


class EnvironmentIterator:  # Todo: Make it raise an error when iteration is finished instead of failing silently
    """
    
    Usage:
    >>> import logging
    >>> env = Environment(name='some_name')
    >>> iterator = EnvironmentIterator(env, num_episodes=500, render_env=True, log=logging.getLogger(__name__))
    >>> for iter_no in iterator:
    ...     action = env.actions.sample()
    ...     iterator.send(action=action)
    ...     print(iterator.state, iterator.new_state, iterator.step_reward, iterator.done)
    """
    def __init__(self, env, num_episodes, render_env=False, log=None):
        self.env = env
        self.num_episodes = num_episodes
        self.episode_id = 0
        self.total_episode_reward = None
        self.episode_rewards = []
        self.render_env = render_env
        self.new_state, self.step_reward, self.done = [None] * 3
        self.log = log
        self._reset()

    def __iter__(self):
        for self.iter_no in range(self.num_episodes):
            if self.new_state is not None:
                try:
                    self.state = self.new_state[:]
                except TypeError:
                    # Must be a number instead of an array
                    self.state = self.new_state
            if self.render_env:
                self.env.render()
            yield self.iter_no
            if self.done:
                if self.log is not None:
                    self.log.info("Episode %d with iteration %d reward is: %s",
                                  self.episode_id, self.iter_no, self.total_episode_reward)
                self._reset()

    def send(self, action):
        self.new_state, self.step_reward, self.done, _ = self.env.step(action)
        self.total_episode_reward += self.step_reward

    def _reset(self):
        if self.total_episode_reward is not None:
            self.episode_rewards.append(self.total_episode_reward)

        self.state = self.env.reset()
        self.total_episode_reward = 0
        self.episode_id += 1
        self.new_state = None


class MDP:
    """
    Base class which represents a Markov Decision Process.
    
    A Markov Decision Process is characterized by states and actions.
    """

    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        self.current_state = None

    def transition(self, action):
        raise NotImplementedError  # should return a state and a reward

    def reward(self, action):
        """
        Tells you the reward for an action without changing the current state.
        :param action: 
        :return: Reward
        """
        raise NotImplementedError  # just returns a reward
