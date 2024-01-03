import numpy
import cv2

import gymnasium as gym

class RoomsEnv:

    def __init__(self, envs_count, height, width, size = 16, noise = 0.0):

        numpy.random.seed(seed=0)

        self.envs_count = envs_count
        self.height     = height
        self.width      = width
        self.size       = size
        self.noise      = noise

        self.position   = numpy.ones((envs_count, 2), dtype=int)*size//2
        self.map, self.dynamic_items        = self._init_map(height, width, size)

        #lifetime explored rooms flags
        self.explored_rooms  = numpy.zeros((height, width), dtype=int)

        #episodic only explored rooms
        self.episodic_explored_rooms = numpy.zeros((envs_count, height, width), dtype=int)


        self.explored_map               = numpy.zeros((height*size, width*size), dtype=int)
        self.episodic_explored_map_tmp  = numpy.zeros((envs_count, height*size, width*size), dtype=int)
        self.episodic_explored_map      = numpy.zeros((envs_count, height*size, width*size), dtype=int)

        self.steps = numpy.zeros((envs_count, ), dtype=int)
        self.max_steps = height*width*size*4


        self.explored_rooms_count          = 0
        self.episodic_explored_rooms_count = numpy.zeros(envs_count, dtype=int)

        self.observation_space  = gym.spaces.Box(low=0.0, high=1.0, shape=(3, size, size), dtype=numpy.float32)
        self.action_space       = gym.spaces.Discrete(5)

    def __len__(self):
        return self.envs_count


    def render(self, env_id):
        im = self._get_state()
        im = im[env_id]
        im = numpy.moveaxis(im, 0, 2)
        im = cv2.resize(im, (256, 256), interpolation=cv2.INTER_NEAREST)

        
        #im = numpy.log(1.0 + self.explored_map)
        #im = im/(im.max() + 10**-6)

        cv2.imshow("env render", im)
        cv2.waitKey(1)


        


    def reset(self, env_id):
        self.explored_rooms_count                  = self.explored_rooms.sum()
        self.episodic_explored_rooms_count[env_id] = self.episodic_explored_rooms[env_id].sum()
        

        self.position[env_id]                = self.size//2
        self.episodic_explored_rooms[env_id] = 0

        self.episodic_explored_map[env_id]        = self.episodic_explored_map_tmp[env_id]
        self.episodic_explored_map_tmp[env_id]    = 0

        self.state = self._get_state()

        self.steps[env_id] = 0
        
        return self.state[env_id], False
    
    def step(self, actions):

        dy = numpy.zeros(self.envs_count)
        dx = numpy.zeros(self.envs_count)

        dy+= 1*(actions == 1)
        dx+= 1*(actions == 2)
        dy-= 1*(actions == 3)
        dx-= 1*(actions == 4) 

        self.position[:, 0] = numpy.clip(self.position[:, 0] + dy, 0, self.height*self.size-1)
        self.position[:, 1] = numpy.clip(self.position[:, 1] + dx, 0, self.width*self.size-1)
        
        self.state = self._get_state()

        room_y = self.position[:, 0]//self.size
        room_x = self.position[:, 1]//self.size

        for e in range(self.envs_count):
            self.explored_rooms[room_y[e], room_x[e]] = 1
            self.explored_map[self.position[e, 0], self.position[e, 1]]+= 1
            self.episodic_explored_map_tmp[e, self.position[e, 0], self.position[e, 1]]+= 1


        self.episodic_explored_rooms[range(self.envs_count), room_y, room_x] = 1

        #print(self.episodic_explored_rooms.sum(axis=(1, 2)))

        rewards = 1.0*(self.episodic_explored_rooms.sum(axis=(1, 2)) == self.width*self.height)

        self.steps+= 1

        colisions = self._get_colision()
        dones     = numpy.logical_or((self.steps >= self.max_steps), colisions)
        dones     = numpy.logical_or(dones, rewards > 0.0)
        

        infos   = self._get_infos()


        return self.state, rewards, dones, False, infos
    

    def save(self, path):
        with open(path + "explored_map.npy", "wb") as f:
            numpy.save(f, self.explored_map)
              

    def _init_map(self, height, width, size):
        result  = numpy.zeros((height, width, 3, size, size), dtype=numpy.float32)

        

        dynamic_items_result = numpy.zeros((height, width, size, size), dtype=int)

        colors = []
        colors.append([1.0, 0.0, 0.0])
        colors.append([0.0, 1.0, 0.0])
        colors.append([0.0, 0.0, 1.0])
        colors.append([1.0, 1.0, 0.0])
        colors.append([0.0, 1.0, 0.0])
        colors.append([1.0, 0.0, 1.0])
        colors.append([0.0, 1.0, 1.0])

        for _ in range((height*width*size*size)//10):
            h = numpy.random.randint(0, height)
            w = numpy.random.randint(0, width)
            y = numpy.random.randint(0, size)
            x = numpy.random.randint(0, size)
            dynamic_items_result[h, w, y, x] = numpy.random.randint(1, 64)


        for h in range(height):
            for w in range(width):
                id = numpy.random.randint(len(colors))
                color = colors[id]
                color = numpy.expand_dims(color, 1)
                color = numpy.expand_dims(color, 2)

                result[h][w] = color/2.0

                for i in range(self.size):
                    id = numpy.random.randint(len(colors))
                    color = colors[id]
                   

                    y = numpy.random.randint(0, self.size)
                    x = numpy.random.randint(0, self.size)
                    result[h, w, :, y, x] = color

                #add obstacles
                for i in range(int(self.size**0.5)):
                    y = numpy.random.randint(0, self.size)
                    x = numpy.random.randint(0, self.size)
                    result[h, w, :, y, x] = 0

        return result, dynamic_items_result
    
    def _get_state(self):
        result = numpy.zeros((self.envs_count, 3, self.size, self.size), dtype=numpy.float32)

        room_y = self.position[:, 0]//self.size
        room_x = self.position[:, 1]//self.size
        ofs_y  = self.position[:, 0]%self.size
        ofs_x  = self.position[:, 1]%self.size
        
        for e in range(self.envs_count):
            #fill with correct room id
            result[e] = self.map[room_y[e], room_x[e]].copy()
            result[e]+= self.noise*numpy.random.randn(3, self.size, self.size)

            #random invert colors
            s = self.steps[e]

            mod = (self.dynamic_items[room_y[e], room_x[e]]+1)
            change = 1.0*(s%mod > mod//2)
            change = numpy.expand_dims(change, 0)
            result[e] =  (1.0 - change)*result[e] + change*(1.0 - result[e])
            
            
            #put player position (white, value 1)
            result[e, :, ofs_y[e], ofs_x[e]] = 1.0

           

        return result
    
    def _get_infos(self):

        result = [] 

        episodic_explored_area = (self.episodic_explored_map > 0.0).mean()

        for e in range(self.envs_count): 
            info = {}
            info["explored_rooms_count"]          = self.explored_rooms_count
            info["episodic_explored_rooms_count"] = self.episodic_explored_rooms_count[e]
            info["episodic_explored_area"]        = round(episodic_explored_area, 5)
            result.append(info)

        return result
    

    def _get_colision(self):

        room_y = self.position[:, 0]//self.size
        room_x = self.position[:, 1]//self.size
        ofs_y  = self.position[:, 0]%self.size
        ofs_x  = self.position[:, 1]%self.size

        result = numpy.zeros(self.envs_count, dtype=bool)
        
        for e in range(self.envs_count):
            x = self.map[room_y[e], room_x[e], :, ofs_y[e], ofs_x[e]]
            x = x.sum()

            if x < 10**-6:
                result[e] = True

        return result
    
import time

if __name__ == "__main__":
    envs_count = 128
    
    env = RoomsEnv(128, 16, 16, 16, 0.0)
    
    for e in range(envs_count):
        env.reset(e)


    fps = 0.0
    k   = 0.99
    step = 0
    while True:
        env.render(10)

        time_start = time.time()
        actions = numpy.random.randint(0, 5, (envs_count, ))
        state, rewards, dones, _, infos = env.step(actions)
        time_stop = time.time()

        fps = k*fps + (1.0 - k)*envs_count/(time_stop - time_start)

        
        if step%1000 == 0:
            print(step, "fps = ", round(fps, 2), round(fps, 2)/envs_count, infos[10], rewards[10])
        step+= 1

        for e in range(len(dones)):
            if dones[e]:
                env.reset(e)
