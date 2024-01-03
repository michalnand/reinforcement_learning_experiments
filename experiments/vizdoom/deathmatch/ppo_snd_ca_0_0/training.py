import doom_wrapper
import RLAgents

import src.model_ppo          as ModelPPO
import src.model_predictor    as ModelPredictor
import src.model_target       as ModelTarget
import src.config             as Config
  
path = "./" 
 
config  = Config.Config()  
 
envs = RLAgents.MultiEnvParallelOptimised("deathmatch.wad", doom_wrapper.DoomWrapper, config.envs_count, threads_count=16)
 
agent = RLAgents.AgentPPOSNDCA(envs, ModelPPO, ModelPredictor, ModelTarget, config)
 
max_iterations = 1000000 
  

trainig = RLAgents.TrainingIterations(envs, agent, max_iterations, path, 128)
trainig.run() 
 