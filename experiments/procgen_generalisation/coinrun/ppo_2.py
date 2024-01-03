import RLAgents

import models.ppo_2.src.model_ppo          as ModelPPO
import models.ppo_2.src.config             as Config

#torch.cuda.set_device("cuda:1")


path = "models/ppo_2/" 
 
config  = Config.Config()  

#config.envs_count = 1  

envs = RLAgents.MultiEnvSeq("procgen-coinrun-v0", RLAgents.WrapperSparseExplorationHard, config.envs_count)

agent = RLAgents.AgentPPO(envs, ModelPPO, config)
 
max_iterations = 1000000 
  

trainig = RLAgents.TrainingIterations(envs, agent, max_iterations, path, 128)
trainig.run() 


'''
agent.load(path)
agent.disable_training()


while True:
    reward, done, info = agent.main()
    if done:
        break
'''