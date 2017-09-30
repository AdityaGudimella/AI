import logging

import attrdict
import gym
import numpy as np

from src.agents import QTable, QTableParams
from src.environments import EnvironmentIterator

env = gym.make('FrozenLake-v0')
agent = QTable(action_dim=env.action_space.n, state_dim=env.observation_space.n, params=QTableParams())

iterator = EnvironmentIterator(env, num_episodes=20000, render_env=True, log=logging.getLogger(__name__))
for iter_no in iterator:
    action = agent.act(iterator.state)
    iterator.send(action=action)
    train_dict = attrdict.AttrDict(dict(state=iterator.state,
                                        action=action,
                                        new_state=iterator.new_state,
                                        reward=iterator.step_reward))
    agent.train(train_dict)
    state = iterator.new_state
    if iter_no % 50 == 0:
        if len(iterator.episode_rewards) > 0:
            print(iterator.episode_rewards)
