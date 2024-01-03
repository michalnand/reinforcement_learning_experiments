import RLAgents

import models.ppo_snd_ca_0.src.model_ppo          as ModelPPO
import models.ppo_snd_ca_0.src.model_predictor    as ModelPredictor
import models.ppo_snd_ca_0.src.model_target       as ModelTarget
import models.ppo_snd_ca_0.src.config             as Config

#torch.cuda.set_device("cuda:1")


path = "models/ppo_snd_ca_0/" 

config  = Config.Config() 

#config.envs_count = 1  

envs = RLAgents.MultiEnvSeq("procgen-coinrun-v0", RLAgents.WrapperSparseExplorationHard, config.envs_count)

agent = RLAgents.AgentPPOSNDCA(envs, ModelPPO, ModelPredictor, ModelTarget, config)
 
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