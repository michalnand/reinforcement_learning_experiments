import os
import numpy
import vizdoom
import gymnasium as gym

from PIL import Image
import cv2




class CreateDoomGame(gym.Wrapper ):
    def __init__(self, scenario_name = "basic.wad", map_name = "map01", render = False, episode_max_steps=4*4500):
        super(CreateDoomGame, self).__init__(None)

        game = vizdoom.DoomGame()

        # Now it's time for configuration!
        # load_config could be used to load configuration instead of doing it here with code.
        # If load_config is used in-code configuration will also work - most recent changes will add to previous ones.
        # game.load_config("../../scenarios/basic.cfg")

        # Sets path to additional resources wad file which is basically your scenario wad.
        # If not specified default maps will be used and it's pretty much useless... unless you want to play good old Doom.
        game.set_doom_scenario_path(os.path.join(vizdoom.scenarios_path, scenario_name))


        # Sets map to start (scenario .wad files can contain many maps).
        game.set_doom_map(map_name)

        # Sets resolution. Default is 320X240
        game.set_screen_resolution(vizdoom.ScreenResolution.RES_320X240)

        # Sets the screen buffer format. Not used here but now you can change it. Default is CRCGCB.
        game.set_screen_format(vizdoom.ScreenFormat.RGB24)

        # Enables depth buffer (turned off by default).
        game.set_depth_buffer_enabled(True)

        # Enables labeling of in-game objects labeling (turned off by default).
        game.set_labels_buffer_enabled(True)

        # Enables buffer with a top-down map of the current episode/level (turned off by default).
        game.set_automap_buffer_enabled(True)

        # Enables information about all objects present in the current episode/level (turned off by default).
        game.set_objects_info_enabled(True)

        # Enables information about all sectors (map layout/geometry, turned off by default).
        game.set_sectors_info_enabled(True)

        # Sets other rendering options (all of these options except crosshair are enabled (set to True) by default)
        game.set_render_hud(True)
        game.set_render_minimal_hud(False)  # If hud is enabled
        game.set_render_crosshair(False)
        game.set_render_weapon(True) 
        game.set_render_decals(False)  # Bullet holes and blood on the walls
        game.set_render_particles(False)
        game.set_render_effects_sprites(False)  # Like smoke and blood
        game.set_render_messages(False)  # In-game text messages
        game.set_render_corpses(False)
        game.set_render_screen_flashes(
            True
        ) 

        # Causes episodes to finish after 200 tics (actions)
        game.set_episode_timeout(200)

        # Makes episodes start after 10 tics (~after raising the weapon)
        game.set_episode_start_time(10)

        # Makes the window appear (turned on by default)
        game.set_window_visible(render)

        # Turns on the sound. (turned off by default)
        game.set_sound_enabled(False)

        # Because of some problems with OpenAL on Ubuntu 20.04, we keep this line commented,
        # the sound is only useful for humans watching the game.

        # Turns on the audio buffer. (turned off by default)
        # If this is switched on, the audio will stop playing on device, even with game.set_sound_enabled(True)
        # Setting game.set_sound_enabled(True) is not required for audio buffer to work.
        # game.set_audio_buffer_enabled(True)
        # Because of some problems with OpenAL on Ubuntu 20.04, we keep this line commented.

        # Sets the living reward (for each move) to -1
        #game.set_living_reward(-1)

        buttons = []

        buttons.append(vizdoom.Button.ALTATTACK)
        buttons.append(vizdoom.Button.ATTACK)
        
        buttons.append(vizdoom.Button.TURN_LEFT)
        buttons.append(vizdoom.Button.TURN_RIGHT)
        buttons.append(vizdoom.Button.MOVE_LEFT)
        buttons.append(vizdoom.Button.MOVE_RIGHT)
        buttons.append(vizdoom.Button.MOVE_FORWARD)
        buttons.append(vizdoom.Button.MOVE_BACKWARD)
        buttons.append(vizdoom.Button.JUMP)
        buttons.append(vizdoom.Button.USE)

        buttons.append(vizdoom.Button.SELECT_NEXT_WEAPON)
        buttons.append(vizdoom.Button.SELECT_PREV_WEAPON)

        game.set_available_buttons(buttons)
        

        game.set_episode_timeout(episode_max_steps)

        # Makes episodes start after 10 tics (~after raising the weapon)
        game.set_episode_start_time(10)

        # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
        game.set_mode(vizdoom.Mode.PLAYER)

        game.init()
        actions = numpy.eye(len(buttons), dtype=bool)


        self.game    = game
        self.actions = actions

        self.observation_shape = (240, 320, 3)
        self.observation_space = gym.spaces.Box(    low = numpy.zeros(self.observation_shape), 
                                                    high = numpy.ones(self.observation_shape),
                                                    dtype = numpy.uint8)
    
        self.action_space = gym.spaces.Discrete(len(buttons),)

        self.reward_prev = 0
        self.reward_now  = 0

    def reset(self):
        self.game.respawn_player()
        state = self.game.get_state().screen_buffer

        self.reward_prev = self.game.get_total_reward()
        self.reward_now  = self.reward_prev

        info  = {}

        return state, info
    

    def step(self, action):
        

        a = self.actions[action]

        self.game.make_action(a)

        state = self.game.get_state()

        if state is not None:
            state = self.game.get_state().screen_buffer
        else:
            state = numpy.zeros(self.observation_shape, dtype=numpy.uint8)

        self.reward_prev = self.reward_now
        self.reward_now  = self.game.get_total_reward()

        reward = self.reward_now - self.reward_prev

        done      = self.game.is_player_dead() or self.game.is_episode_finished()
        
        truncated = False 
        info      = {}


        return state, reward, done, truncated, info

    def close(self):
        self.game.close()

class BasicWrapper(gym.Wrapper ):
    def __init__(self, env, action_repeat, max_steps, env_name):
        super(BasicWrapper, self).__init__(env)

        self.env            = env
        self.action_repeat  = action_repeat
        self.max_steps      = max_steps
        self.steps          = 0
        self.env_name       = str(env_name)

        self.reward_sum = 0
        self.reward_sum_tmp = 0

    def step(self, action):
        for i in range(self.action_repeat):
            self.steps+= 1
            observation, reward, done, truncated, info = self.env.step(action)
            if done:
                break
        
        self.reward_sum_tmp+= reward

        if self.steps >= self.max_steps:
            done = True
        
        if done:
            self.reward_sum = self.reward_sum_tmp
            self.reward_sum_tmp = 0

        

        return observation, reward, done, truncated, info
    
    def reset(self, seed = None, options = None):
        observation, info = self.env.reset()

        self.steps = 0

        return observation, info
    


class AugmentationWrapper(gym.Wrapper ):
    def __init__(self, env, height = 96, width = 96):
        super(AugmentationWrapper, self).__init__(env)
        self.height = height
        self.width  = width
       

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        state = self._update_state(state)  
        return state, reward, done, truncated, info
    
    
    def reset(self, seed = None, options = None):
        self._create_augmentation()

        state, info = self.env.reset()
        state = self._update_state(state)

        

        return state, info
    

    def _update_state(self, state):
        img = Image.fromarray(state)
        img = img.resize((self.height, self.width))
        img = numpy.array(img, dtype=numpy.float32)

        #random negative
        if self.red_flip:
            img[:, :, 0] = 255 - img[:, :, 0]
        if self.green_flip:
            img[:, :, 1] = 255 - img[:, :, 1]
        if self.blue_flip:
            img[:, :, 2] = 255 - img[:, :, 2]

        #random color change
        img = img[:, :, self.color_permutation]
        
        #random brightness
        img[:, :, 0]*= self.red_brightness
        img[:, :, 1]*= self.green_brightness
        img[:, :, 2]*= self.blue_brightness

        img = numpy.clip(img, 0, 255)

        return img
    
    def _create_augmentation(self):
        self.red_flip   = numpy.random.rand() < 0.5
        self.green_flip = numpy.random.rand() < 0.5
        self.blue_flip  = numpy.random.rand() < 0.5

        self.color_permutation = numpy.random.permutation(3)

        
        brigtness_min = 0.2
        brightness_max= 1.5
        
        x = numpy.random.rand()
        self.red_brightness   = (1-x)*brigtness_min + x*brightness_max

        x = numpy.random.rand()
        self.green_brightness = (1-x)*brigtness_min + x*brightness_max

        x = numpy.random.rand()
        self.blue_brightness  = (1-x)*brigtness_min + x*brightness_max
        


class StateWrapper(gym.Wrapper ):
    def __init__(self, env, height = 96, width = 96, frame_stacking = 4, normalise = False):
        super(StateWrapper, self).__init__(env)
        self.height = height
        self.width  = width
        self.frame_stacking = frame_stacking
        self.normalise      = normalise

        self.state_shape = (self.frame_stacking*3, self.height, self.width)
        self.dtype  = numpy.float32

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.state_shape, dtype=self.dtype)
        self.state = numpy.zeros(self.state_shape, dtype=self.dtype)

        

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)

        self._update_state(state)
        
        return self.state, reward, done, truncated, info
    
    
    def reset(self, seed = None, options = None):
        state, info = self.env.reset()
        
        self.state[:] = 0
        self._update_state(state)

        return self.state, info
    

    def _update_state(self, state):
        state = numpy.array(state, dtype=numpy.uint8)
        img = Image.fromarray(state)
        img = img.resize((self.width, self.height))
        
        s_new = numpy.array(img, dtype=numpy.float32)/255.0
        s_new = numpy.moveaxis(s_new, 2, 0)
        
        

        if self.normalise: 
            s_new = (s_new - s_new.mean())/(s_new.std() + 10**-5)

        self.state      = numpy.roll(self.state, 3, axis=0)
        self.state[0:3] = s_new

        '''
        #state render for debug
        im = cv2.resize(s_new, (512, 512), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("visualisation", im)
        cv2.waitKey(1)
        '''
        
        
        

       

     


class RewardWrapper(gym.Wrapper ):
    def __init__(self, env):
        super(RewardWrapper, self).__init__(env)

        self.env = env

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)

        if reward > 0.0:
            reward = 1.0
        elif done and reward <= 0.0:
            reward = -1.0
        else:
            reward = 0.0
        
        return state, reward, done, truncated, info
    

    def reset(self):
        return self.env.reset()
    
 



def DoomWrapper(scenario_name = "basic.wad", height = 64, width = 128, frame_stacking = 2, render = False):

    deatch_math_maps = []
    deatch_math_maps.append("map01")
    deatch_math_maps.append("map02")
    deatch_math_maps.append("map03")
    deatch_math_maps.append("map05")
    deatch_math_maps.append("map06")
    deatch_math_maps.append("map08")
    deatch_math_maps.append("map10")
    deatch_math_maps.append("map11")
    deatch_math_maps.append("map13")
    deatch_math_maps.append("map14")
    deatch_math_maps.append("map15")
    deatch_math_maps.append("map17")
    deatch_math_maps.append("map18")
    deatch_math_maps.append("map19")
    deatch_math_maps.append("map22")
    deatch_math_maps.append("map23")
    deatch_math_maps.append("map24")
    deatch_math_maps.append("map25")
    deatch_math_maps.append("map27")
    deatch_math_maps.append("map30")
    deatch_math_maps.append("map31")
    deatch_math_maps.append("map32")
    
    
    map_id = 0 # numpy.random.randint(0, len(deatch_math_maps))
    print("maps count    = ", len(deatch_math_maps))
    print("selecting map = ", map_id)

    env = CreateDoomGame(scenario_name, map_name = deatch_math_maps[map_id], render = render)

    env = BasicWrapper(env, 4, 4500, env)
    env = StateWrapper(env, height, width, frame_stacking, False)
    env = RewardWrapper(env) 

    return env


def DoomWrapperRender(env, height = 64, width = 128, frame_stacking = 2):
    return DoomWrapper(env, height, width, frame_stacking, True)
    