import RLAgents

import models.ppo_ce_2_0.src.model_ppo          as ModelPPO
import models.ppo_ce_2_0.src.model_im           as ModelIM
import models.ppo_ce_2_0.src.config             as Config

#torch.cuda.set_device("cuda:0")
  
path = "models/ppo_ce_2_0/"
 
config  = Config.Config() 

#config.envs_count = 1
 
envs = RLAgents.MultiEnvParallelOptimised("PitfallNoFrameskip-v4", RLAgents.WrapperMontezuma, config.envs_count)
#envs = RLAgents.MultiEnvSeq("PitfallNoFrameskip-v4", RLAgents.WrapperMontezuma, config.envs_count, True)
#envs = RLAgents.MultiEnvSeq("PitfallNoFrameskip-v4", RLAgents.WrapperMontezumaVideo, config.envs_count)
 
agent = RLAgents.AgentPPOCE(envs, ModelPPO, ModelIM, config)
 
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