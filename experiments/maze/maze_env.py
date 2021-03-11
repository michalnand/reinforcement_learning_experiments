import numpy
import gym
import time

import cv2


class MazeEnv:

    def __init__(self, envs_count = 64, size = 19):
        self.envs_count = envs_count
        self.size       = size

        self.observation_space 		= gym.spaces.Box(low=-1, high=1, shape=(1, self.size, self.size))
        self.action_space 	        = gym.spaces.Discrete(4)

        self.base, self.base_positions       = self._create_base()

        self.reset()

    def reset(self, env_id = -1):
        if env_id == -1:
            self.mazes          = numpy.zeros((self.envs_count, self.size, self.size), dtype=numpy.float32)
            self.positions      = numpy.ones((self.envs_count, 2), dtype=int)
            self.steps          = numpy.zeros((self.envs_count), dtype=int)
            for i in range(self.envs_count):
                self.positions[env_id]      = self._random_start()
                self.mazes[i] = self._create_maze()
        else:
            self.mazes[env_id]          = self._create_maze()

            self.positions[env_id]      = self._random_start()
            self.steps[env_id]          = 0

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

        d_position+= [1, 0]*(actions == 0)
        d_position+= [-1, 0]*(actions == 1)
        d_position+= [0, 1]*(actions == 2)
        d_position+= [0, -1]*(actions == 3)

        position_new = self.positions + d_position

        for i in range(self.envs_count):
            y = position_new[i][0]
            x = position_new[i][1]

            if self.mazes[i][y][x] > -0.999:
                self.positions[i] = position_new[i]

            if y == self.size-2 and x == self.size-2:
                rewards[i]      = 1.0
                dones[i]        = True

            if self.steps[i] >= 4*self.size*self.size:
                rewards[i]      = -1
                dones[i]        = True
                self.steps[i]   = 0
                
        return self._update_observations(), rewards, dones, None


    def render(self, env_id = -1):

        if env_id != -1:
            image = self._get_maze_image(env_id, 256)
        else:
            grid_size   = int(self.envs_count**0.5)
            image_size  = 200

            image       = numpy.zeros((grid_size*image_size, grid_size*image_size, 3), numpy.uint8)

            for y in range(grid_size):
                for x in range(grid_size):
                    im  = self._get_maze_image(y*grid_size + x, image_size)

                    y_ofs = y*image_size
                    x_ofs = x*image_size

                    image[0 + y_ofs:image_size + y_ofs, 0 + x_ofs:image_size + x_ofs] = im


        window_name = "Maze - " + self.__class__.__name__ + " " + str(env_id)
        
        cv2.imshow(window_name, image) 
        cv2.waitKey(1)

    def _get_maze_image(self, env_id, image_size):
        image = numpy.zeros((image_size, image_size,3), numpy.uint8)

        element_size = int(image_size/self.size)

        for y in range(self.size):
            for x in range(self.size):
                el = self.mazes[env_id][y][x]

                cy = int(y*element_size)
                cx = int(x*element_size)

                if el < 0.0:
                    image = cv2.rectangle(image, (cx, cy), (cx + element_size, cy + element_size), (255, 0, 0))

                if y == self.positions[env_id][0] and x == self.positions[env_id][1]:
                    image = cv2.circle(image, (cx + element_size//2, cy + element_size//2), int(0.7*0.5*element_size), (0, 0, 255), 2)

                if y == self.size-2 and x == self.size-2:
                    image = cv2.circle(image, (cx + element_size//2, cy + element_size//2), int(0.7*0.5*element_size), (0, 255, 0), 2)

        return image




    def _create_maze(self):
        maze = self.base.copy()

        bases = self.base_positions.copy()
        
        while len(bases) > 0:
            base_idx    = numpy.random.randint(len(bases))
            y, x        = bases[base_idx]
            
            way         = numpy.random.randint(4)

            if way == 0:
                dx = 1
                dy = 0
            elif way == 1:
                dx = -1
                dy = 0
            elif way == 2:
                dx = 0
                dy = 1
            elif way == 3:
                dx = 0
                dy = -1

            maze[y+dy][x+dx] = -1

            del bases[base_idx]

        return maze


    def _create_base(self):
        base = numpy.zeros((self.size, self.size), dtype=numpy.float32)

        for j in range(self.size):
            base[0][j]      = -1
            base[-1][j]     = -1
            base[j][0]      = -1
            base[j][-1]     = -1
        
        base_positions = []
        for j in range(self.size):
            for i in range(self.size):
                if i > 0 and j > 0 and i < self.size-1 and j < self.size-2:
                    if (j%2) == 0 and (i%2) == 0:
                        base[j][i] = -1
                        base_positions.append([j, i])

        return base, base_positions


    def _update_observations(self):
        self.observations = self.mazes.copy().astype(numpy.float32)

        for i in range(self.envs_count):
            y = self.positions[i][0]
            x = self.positions[i][1]
            self.observations[i][y][x] = 1.0

        self.observations = self.observations.reshape((self.envs_count, 1, self.size, self.size))

        return self.observations

    def _random_start(self):
        v = numpy.random.randint(3)
        if v == 0:
            return [1, 1]
        elif v == 1:
            return [1, self.size-2]
        elif v == 2:
            return [self.size-2, 1]


if __name__ == "__main__":

    envs_count  = 16

    env = MazeEnv(envs_count)

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

        #env.render(-1)