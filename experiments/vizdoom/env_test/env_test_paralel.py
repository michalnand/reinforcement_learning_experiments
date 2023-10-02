import numpy
import gymnasium
from vizdoom import gymnasium_wrapper

import doom_wrapper

import RLAgents


if __name__ == "__main__":
    envs_count = 8

    #env = gymnasium.make("VizdoomBasic-v0")
    #env = gymnasium.make("VizdoomDefendLine-v0")
    #env = gymnasium.make("VizdoomCorridor-v0")
    #env = gymnasium.make("VizdoomMyWayHome-v0")
    #env =  doom_wrapper.DoomGenericWrapper(env, render = False)


    #envs = RLAgents.MultiEnvParallelOptimised("basic.wad", doom_wrapper.DoomWrapper, envs_count, threads_count=2)
    #envs = RLAgents.MultiEnvParallelOptimised("defend_the_line.wad", doom_wrapper.DoomWrapperRender, envs_count, threads_count=2)
    envs = RLAgents.MultiEnvParallelOptimised("deathmatch.wad", doom_wrapper.DoomWrapperRender, envs_count, threads_count=8)


    for i in range(envs_count):
        envs.reset(i)



    actions_count = envs.action_space.n

    rewards_sum = numpy.zeros(envs_count)
    dones_sum   = numpy.zeros(envs_count)

    steps = 0

    while True:    
        actions = numpy.random.randint(low = 0, high = actions_count, size=(envs_count, ))
        observations, rewards, dones, _, _ = envs.step(actions)
        

        rewards_sum+= rewards

        for i in range(envs_count):
            if dones[i]:
                envs.reset(i)
                dones_sum[i]+= 1
        
        if steps%100 == 0:
            print("reward per spisode = ", dones_sum, rewards_sum/(dones_sum + 10**-10))

        steps+= 1
        


        