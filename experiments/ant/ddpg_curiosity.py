import gym
import pybullet_envs
import numpy
import time

import RLAgents

import models.ddpg_curiosity.model.src.model_critic             as ModelCritic
import models.ddpg_curiosity.model.src.model_actor              as ModelActor
import models.ddpg_curiosity.model.src.model_forward            as ModelForward
import models.ddpg_curiosity.model.src.model_forward_target     as ModelForwardTarget
import models.ddpg_curiosity.model.src.config                   as Config

path    = "models/ddpg_curiosity/model/"

env     = gym.make("AntBulletEnv-v0")

agent   = RLAgents.AgentDDPGCuriosity(env, ModelCritic, ModelActor, ModelForward, ModelForwardTarget, Config)

max_iterations = 4*(10**6)
trainig = RLAgents.TrainingIterations(env, agent, max_iterations, path, 1000)
trainig.run()

'''
agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main()
    time.sleep(0.01)
'''