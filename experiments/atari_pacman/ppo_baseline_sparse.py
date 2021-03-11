import gym
import numpy
import time

import RLAgents

import models.ppo_baseline_sparse.src.model            as Model
import models.ppo_baseline_sparse.src.config           as Config


path = "models/ppo_baseline_sparse/"

config  = Config.Config()

envs = RLAgents.MultiEnvSeq("MsPacmanNoFrameskip-v4", RLAgents.WrapperAtariSparseRewards, config.actors)

agent = RLAgents.AgentPPO(envs, Model, Config) 

max_iterations = 1*(10**6) 

trainig = RLAgents.TrainingIterations(envs, agent, max_iterations, path, 1000)
trainig.run() 

'''
agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main()

    envs.render(0)
    time.sleep(0.01)
'''