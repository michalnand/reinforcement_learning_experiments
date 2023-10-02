import RLAgents
import time
import numpy

import gymnasium as gym

from PIL import Image


class VectorisedEnv:

    def __init__(self, env_name, envs_count, height = 96, width = 96, frame_stacking = 4):

        self.envs = []

        self.envs_count = envs_count

        self.height         = height
        self.width          = width
        self.frame_stacking = frame_stacking

        self.state_shape    = (self.frame_stacking, self.height, self.width)

        for i in range(self.envs_count):
            env = gym.make(env_name)
            self.envs.append(env)


    def step(self, actions):

        states   = numpy.zeros((self.envs_count, ) + self.state_shape, dtype=numpy.float32)
        rewards  = numpy.zeros((self.envs_count, ), dtype=numpy.float32)
        dones    = numpy.zeros((self.envs_count, ), dtype=int)
        infos    = []


        for i in range(self.envs_count):

            for j in range(4):
                state, reward, done, _, info = self.envs[i].step(actions[i])
                if done:
                    break

           
            img = Image.fromarray(state)
            img = img.convert('L')
            img = img.resize((self.height, self.width))
            state = numpy.array(img, dtype=numpy.float32)/255.0
            

            states[i][:]   = state
            rewards[i]  = reward
            dones[i]    = done
            infos.append(info)

      
        return states, rewards, dones, None, infos
    
    def reset(self, env_id):
        return self.envs[env_id].reset()







if __name__ == "__main__":

    envs_count = 128

    envs = RLAgents.MultiEnvSeq("MontezumaRevengeNoFrameskip-v4", RLAgents.WrapperMontezuma, envs_count)

    envs = VectorisedEnv("MontezumaRevengeNoFrameskip-v4", envs_count)


    for i in range(envs_count):
        envs.reset(i)


    fps = 0.0
    k   = 0.98

    steps = 0

    while True:



        time_start = time.time()
        actions  = numpy.random.randint(0, 18, (envs_count, ))
        state, rewards, dones, _, info = envs.step(actions)
        time_stop = time.time()

        final = numpy.where(dones)[0]

        for f in final:
            envs.reset(f)


        fps = k*fps + (1.0 - k)*1.0*envs_count/(time_stop - time_start)

        if steps%100 == 0:
            print("fps = ", int(fps), int(fps/envs_count))

        steps+= 1