import gym
import pybullet_envs
import numpy
import time

import RLAgents


import models.ppo_baseline.model.src.model  as Model
import models.ppo_baseline.model.src.config as Config

path = "models/ppo_baseline/model/"


config  = Config.Config()
envs    = RLAgents.MultiEnvSeq("AntBulletEnv-v0", None, config.actors)
#envs.render(0)

agent = RLAgents.AgentPPOContinuous(envs, Model, Config)

max_iterations = 1*(10**6)
trainig = RLAgents.TrainingIterations(envs, agent, max_iterations, path, 1000)
trainig.run()

'''
agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main()
    time.sleep(0.01)
'''