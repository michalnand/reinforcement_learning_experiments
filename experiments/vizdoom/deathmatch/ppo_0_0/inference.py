import doom_wrapper
import RLAgents

import src.model_ppo          as ModelPPO
import src.config             as Config


#torch.cuda.set_device("cuda:0")
  
path = "./"
 
config  = Config.Config()  

config.envs_count = 1
 

envs = RLAgents.MultiEnvSeq("deathmatch.wad", doom_wrapper.DoomWrapperRender, config.envs_count)
 
agent = RLAgents.AgentPPO(envs, ModelPPO, config)
   
agent.load(path) 
agent.disable_training()

kills  = 0
dones  = 0

while True:
    reward, done, info = agent.main()

    if reward > 0:
        kills+= 1
        
    if done:
        dones+= 1
        print("kd = ", kills, dones, kills/dones)
