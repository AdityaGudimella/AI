import attrdict
import logging
from numpy import copy


class RLMetrics:
    def __init__(self):
        `
        ""

class Model:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def train(self, num_iters, logger=None, log_stats=False):
        for i in range(num_iters):
            state = self.env.reset()
            done = False
            while not done:
                action = self.agent.act(state)
                new_state, reward, done, _ = self.env.step(action)
                train_dict = attrdict.AttrDict(dict(state=state,
                                                    action=action,
                                                    new_state=new_state,
                                                    reward=reward))
                self.agent.train(train_dict)

                if log_stats:
                    self.stats(logger=logger)

                state = copy(new_state)

    def stats(self, logger: logging.Logger):
        logger.info('Log some stats here after some test passes')
