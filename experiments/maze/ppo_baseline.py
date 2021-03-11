import numpy
import time
from maze_env import *

import RLAgents

import models.ppo_baseline.src.model            as Model
import models.ppo_baseline.src.config           as Config


path = "models/ppo_baseline/"

config  = Config.Config()
envs    = MazeEnv(config.actors)

agent   = RLAgents.AgentPPO(envs, Model, Config) 

max_iterations = 1*(10**6) 

#trainig = RLAgents.TrainingIterations(envs, agent, max_iterations, path, 1000)
#trainig.run() 


agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main()

    envs.render()
    time.sleep(0.01)
