import gym
import numpy
import time

import RLAgents

import models.ppo_curiosity_1.src.model_ppo               as ModelPPO
import models.ppo_curiosity_1.src.model_forward           as ModelForward
import models.ppo_curiosity_1.src.model_forward_target    as ModelForwardTarget
import models.ppo_curiosity_1.src.config                  as Config


path = "models/ppo_curiosity_1/"

config  = Config.Config()

#envs = RLAgents.MultiEnvParallel("MontezumaRevengeNoFrameskip-v4", RLAgents.WrapperMontezuma, config.actors)
envs = RLAgents.MultiEnvSeq("MontezumaRevengeNoFrameskip-v4", RLAgents.WrapperMontezuma, config.actors)

agent = RLAgents.AgentPPOCuriosity(envs, ModelPPO, ModelForward, ModelForwardTarget, Config)

max_iterations = 2*(10**6) 
 
#trainig = RLAgents.TrainingIterations(envs, agent, max_iterations, path, 128)
#trainig.run() 


agent.load(path)
agent.disable_training()
while True:
    reward, done, _ = agent.main()

    envs.render(0)
    time.sleep(0.01)
    if done:
        break
