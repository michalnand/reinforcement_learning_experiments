import numpy
import time
from maze_env import *

import RLAgents

import models.ppo_curiosity.src.model_ppo               as ModelPPO
import models.ppo_curiosity.src.model_forward           as ModelForward
import models.ppo_curiosity.src.model_forward_target    as ModelForwardTarget
import models.ppo_curiosity.src.config                  as Config


path = "models/ppo_curiosity/"

config  = Config.Config()
envs    = MazeEnv(config.actors)
 
agent = RLAgents.AgentPPOCuriosity(envs, ModelPPO, ModelForward, ModelForwardTarget, Config)

max_iterations = 1*(10**6) 

#trainig = RLAgents.TrainingIterations(envs, agent, max_iterations, path, 1000)
#trainig.run() 


agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main()

    envs.render()
    time.sleep(0.01)
