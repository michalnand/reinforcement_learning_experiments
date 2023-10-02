import doom_wrapper
import RLAgents

import src.model_ppo          as ModelPPO
import src.config             as Config


#torch.cuda.set_device("cuda:0")
  
path = "./" 
 
config  = Config.Config()  

#config.envs_count = 1
 
envs = RLAgents.MultiEnvParallelOptimised("deathmatch.wad", doom_wrapper.DoomWrapper, config.envs_count, threads_count=16)
 
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
