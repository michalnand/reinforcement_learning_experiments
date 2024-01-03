import doom_wrapper
import RLAgents

import src.model_ppo          as ModelPPO
import src.config             as Config
  
path = "./"
 
config  = Config.Config()  

config.envs_count = 1
 

envs = RLAgents.MultiEnvSeq("deathmatch_simple.wad", doom_wrapper.DoomWrapperRender, config.envs_count)
#envs = RLAgents.MultiEnvSeq("deathmatch_simple.wad", doom_wrapper.DoomWrapper, config.envs_count)
 
agent = RLAgents.AgentPPO(envs, ModelPPO, config)
   
agent.load(path) 
agent.disable_training()

kills  = 0
dones  = 0

while True:
    reward, done, info = agent.main()
   
    if done:
        kills+= info["frags"]
        dones+= 1

        print("kd = ", kills, dones, kills/dones)
