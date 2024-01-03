from env.rooms_env import *
import RLAgents

import models.ppo_csnd_5_0.src.model_ppo          as ModelPPO
import models.ppo_csnd_5_0.src.model_target       as ModelTarget
import models.ppo_csnd_5_0.src.model_predictor    as ModelPredictor
import models.ppo_csnd_5_0.src.config             as Config

#torch.cuda.set_device("cuda:0")
  
path = "models/ppo_csnd_5_0/"  
 
config  = Config.Config() 

envs = RoomsEnv(config.envs_count, 16, 16, 16, 0.0)

agent = RLAgents.AgentPPOCSND(envs, ModelPPO, ModelTarget, ModelPredictor, config)
 
max_iterations = 1000000 
  

trainig = RLAgents.TrainingNew(envs, agent, max_iterations, path, 128)
trainig.run()


envs.save(path + "trained/")