import doom_wrapper
import RLAgents

import src.model_ppo          as ModelPPO
import src.config             as Config


#torch.cuda.set_device("cuda:0")
  
path = "./"
 
config  = Config.Config()  

config.envs_count = 1
 

envs = RLAgents.MultiEnvSeq("basic.wad", doom_wrapper.DoomWrapperRender, config.envs_count)
 
agent = RLAgents.AgentPPO(envs, ModelPPO, config)
 
max_iterations = 250000 
  


agent.load(path)
agent.disable_training()

while True:
    reward, done, info = agent.main()
