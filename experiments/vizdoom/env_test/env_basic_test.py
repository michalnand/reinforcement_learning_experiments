import numpy
import doom_wrapper
import cv2

if __name__ == "__main__":

   
    #env_name = "basic.wad"
    env_name = "deathmatch.wad"
    #env_name = "multi_deathmatch.wad"
    env =  doom_wrapper.DoomWrapper(env_name, render = False)
    env.reset()

    actions_count = env.action_space.n

    reward_sum = 0
    done_sum   = 0

    while True:    
        actions = numpy.random.randint(low = 0, high = actions_count)
        observation, reward, done, _, _ = env.step(actions)
        
        reward_sum+= reward

       
        if done:
            env.reset()
            done_sum+= 1

            print("reward per spisode = ", done_sum, reward_sum/(done_sum + 10**-10))

        '''
        print("observation = ", observation.mean(), observation.std(), observation.min(), observation.max())
        s = observation[0:3, :, :]
        s = numpy.moveaxis(s, 0, 2)
        s = cv2.cvtColor(s, cv2.COLOR_RGB2BGR)        
        cv2.imshow("state ", s)
        cv2.waitKey(1)
        '''
        
        