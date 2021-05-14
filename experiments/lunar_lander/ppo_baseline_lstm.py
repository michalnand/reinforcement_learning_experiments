import gym
import numpy
import time

import RLAgents

import models.ppo_baseline_lstm.src.model     as Model
import models.ppo_baseline_lstm.src.config    as Config

path = "models/ppo_baseline_lstm/"

config  = Config.Config()

class Wrapper(gym.Wrapper):
    def __init__(self, env, seq_length = 4):
        gym.Wrapper.__init__(self, env) 

        self.obs_shape          = (seq_length, ) + env.observation_space.shape
        self.observation_space  = gym.spaces.Box(low=-1.0, high=1.0, shape=self.obs_shape, dtype=numpy.float32)
        self.observation        = numpy.zeros(self.obs_shape, dtype=numpy.float32)
       
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = reward/100.0
        return self._observation(obs), reward, done, info

    def reset(self):
        self.observation        = numpy.zeros(self.obs_shape)
        return self._observation(self.env.reset())

    def _observation(self, obs):
        self.observation        = numpy.roll(self.observation, -1, 0)
        self.observation[-1]    = obs.copy()

        return self.observation

def WrapperFunc(env):
    env = Wrapper(env)
    return env


envs = RLAgents.MultiEnvSeq("LunarLanderContinuous-v2", WrapperFunc, config.actors)

agent = RLAgents.AgentPPOContinuous(envs, Model, Config)

max_iterations = 100000
trainig = RLAgents.TrainingIterations(envs, agent, max_iterations, path, 100)
trainig.run()

'''
agent.load(path)
agent.disable_training()
while True:
    agent.main()
    envs.render(0)
'''