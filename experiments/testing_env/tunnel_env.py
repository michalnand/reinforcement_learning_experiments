import numpy
import gym
import time

import cv2


class TunnelEnv:

    def __init__(self, envs_count = 16, height = 5, width = 128):
        self.envs_count     = envs_count 
        self.height         = height
        self.width          = width

        self.observation_space 		= gym.spaces.Box(low=-1, high=1, shape=(2, ))
        self.action_space 	        = gym.spaces.Discrete(5)

        self.reset()

    def reset(self, env_id = -1):
        if env_id == -1:
            self.positions = numpy.zeros((self.envs_count, 2))
            for i in range(self.envs_count):
                self.positions[i][0] = 1 + numpy.random.randint(self.height-2)
                self.positions[i][1] = 0

            self.steps     = numpy.zeros(self.envs_count, dtype=int)
            
        else:
            self.positions[env_id][0] = 1 + numpy.random.randint(self.height-2)
            self.positions[env_id][1] = 0
            self.steps[env_id]   = 0

        if env_id != -1:
            obs = self._update_observations()[env_id]
        else:
            obs = self._update_observations()
            
        return obs


    def step(self, actions):
        rewards = numpy.zeros(self.envs_count, dtype=numpy.float32)
        dones   = numpy.zeros(self.envs_count, dtype=bool)

        self.steps+= 1

        d_position = numpy.zeros((self.envs_count, 2), dtype=int)

        actions    = numpy.array(actions).reshape((self.envs_count, 1))

        d_position+= [0, 0]*(actions == 0)
        d_position+= [1, 0]*(actions == 1)
        d_position+= [-1, 0]*(actions == 2)
        d_position+= [0, 1]*(actions == 3)
        d_position+= [0, -1]*(actions == 4)

        positions_new = self.positions + d_position
        

           

        for i in range(self.envs_count):
            y = positions_new[i][0] 
            x = positions_new[i][1]
            if y >= 0 and x >= 0 and y < self.height and x < self.width:
                self.positions[i][0] = y
                self.positions[i][1] = x

            if y <= 0 or y >= self.height-1:
                rewards[i]      = -1.0
                dones[i]        = True
  
            if x == self.width-1 and y == self.height//2:
                rewards[i]      = 1.0
                dones[i]        = True

            if self.steps[i] >= 2*self.height*self.width:
                rewards[i]      = 0.0
                dones[i]        = True
           

                
        return self._update_observations(), rewards, dones, None


    def render(self, env_id = 0):
        for y in range(self.height):
            for x in range(self.width):
                if y == 0 or y == self.height-1:
                    print("X", end="")
                elif x == self.width-1 and y == self.height//2:
                    print("T", end="")
                elif y == self.positions[env_id][0] and x == self.positions[env_id][1]:
                    print("P", end="")
                else:
                    print(".", end="")
                print(" ", end="")
            print()
        print("\n\n\n")

        element_size    = 8
        height          = self.height*element_size
        width           = self.width *element_size
        image           = numpy.zeros((height, width,3), numpy.uint8)

        for y in range(self.height):
            for x in range(self.width):
                cy = int(y*element_size)
                cx = int(x*element_size)

                if y == 0 or y == self.height-1:
                    r, g, b = 255, 0, 0
                elif x == self.width-1 and y == self.height//2:
                    r, g, b = 0, 255, 0
                elif y == self.positions[env_id][0] and x == self.positions[env_id][1]:
                    r, g, b = 0, 0, 255
                else:
                    r, g, b = 0, 0, 0
              
                if r != 0 or g != 0 or b != 0:
                    image = cv2.rectangle(image, (cx, cy), (cx + element_size, cy + element_size), (b, g, r))
           
        window_name = "ENV - " + self.__class__.__name__ + " " + str(env_id)
        
        cv2.imshow(window_name, image) 
        cv2.waitKey(1)

    def _update_observations(self):
        dims   = numpy.zeros(2)
        dims[0] = self.height
        dims[1] = self.width
        result  = self.positions.copy()/dims
        
        return result


   

if __name__ == "__main__":

    envs_count  = 16

    env = TunnelEnv(envs_count)

    actions_count = env.action_space.n

    fps = 0.0
    k   = 0.01

    while True:

        time_start = time.time()
        actions = numpy.random.randint(actions_count, size=envs_count)
        states, rewards, dones, _ = env.step(actions)
        time_stop = time.time()

        for i in range(len(dones)):
            if dones[i]:
                states = env.reset(i)

        fps = (1.0 - k)*fps + k/(time_stop - time_start)

        print("fps = ", round(fps, 2))

        env.render(-1)