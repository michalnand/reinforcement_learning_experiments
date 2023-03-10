import gym
import numpy
import time

import RLAgents

import models.ppo_entropy_sparse.src.model_ppo             as ModelPPO
import models.ppo_entropy_sparse.src.model_forward         as ModelForward
import models.ppo_entropy_sparse.src.model_forward_target  as ModelForwardTarget
import models.ppo_entropy_sparse.src.model_ae              as ModelAutoencoder
import models.ppo_entropy_sparse.src.config                as Config
 

path = "models/ppo_entropy_sparse/"
 
config  = Config.Config()
 
envs = RLAgents.MultiEnvSeq("MsPacmanNoFrameskip-v4", RLAgents.WrapperAtariSparseRewards, config.actors)

agent = RLAgents.AgentPPOEntropy(envs, ModelPPO, ModelForward, ModelForwardTarget, ModelAutoencoder, Config)
 
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