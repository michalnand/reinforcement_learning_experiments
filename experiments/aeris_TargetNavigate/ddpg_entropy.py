import gym
import gym_aeris
import numpy
import time

import RLAgents


import models.ddpg_entropy.model.src.model_critic               as ModelCritic
import models.ddpg_entropy.model.src.model_actor                as ModelActor
import models.ddpg_entropy.model.src.model_forward              as ModelForward
import models.ddpg_entropy.model.src.model_forward_target       as ModelForwardTarget
import models.ddpg_entropy.model.src.model_ae                   as ModelAutoencoder
import models.ddpg_entropy.model.src.config                     as Config

path = "models/ddpg_entropy/model/"

env = gym.make("TargetNavigate-v0", render = False)

agent = RLAgents.AgentDDPGEntropy(env, ModelCritic, ModelActor, ModelForward, ModelForwardTarget, ModelAutoencoder, Config)

max_iterations = 1*(10**6)
trainig = RLAgents.TrainingIterations(env, agent, max_iterations, path, 10000)
trainig.run()

'''
agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main()
    time.sleep(0.01)
'''