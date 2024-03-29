import os
import numpy
import vizdoom
import gymnasium as gym

from PIL import Image
import cv2




class CreateDoomGame(gym.Wrapper ):
    def __init__(self, scenario_name = "basic.wad", map_name = "map01", render = False, episode_max_steps=4*4500, n_bots = 0):
        super(CreateDoomGame, self).__init__(None)

        self.n_bots = n_bots

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

        if self.n_bots > 0:
            game_args = "-host 1 -deathmatch +viz_nocheat 0 +cl_run 1 +name AGENT +colorset 0 +sv_forcerespawn 1 +sv_respawnprotect 1 +sv_nocrouch 1 +sv_noexit 1"

            #game.add_game_args("-host 1 -deathmatch  +viz_nocheat 1 +sv_spawnfarthest 1")
            game.add_game_args(game_args)
            bots = os.path.join(vizdoom.scenarios_path, "bots.cfg")
            game.add_game_args("+viz_bots_path " + bots)

            game.add_game_args("+name AI +colorset 0")

        #game.set_console_enabled(True)


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

        #game.add_game_args("+viz_bots_path ../../scenarios/perfect_bots.cfg")


      

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
        game.set_render_screen_flashes(True) 

        # Causes episodes to finish after 5000 tics (actions)
        game.set_episode_timeout(5000) 

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

       
    def reset(self):
        self.frags      = 0
        self.frags_prev = 0

        self.damage      = 0
        self.damage_prev = 0

        self.health         = 100
        self.health_prev    = 100

        self.prev_x, self.prev_y = self._get_player_position()


        self.episode_frags  = 0
        self.episode_damage = 0



        self.game.new_episode()
        
        state = self.game.get_state().screen_buffer

        info  = self._make_info()

        self.game.send_game_command('removebots')
        for i in range(self.n_bots):
            self.game.send_game_command('addbot')

        self.game.respawn_player()
        
        return state, info
    

    def step(self, action):
        a = self.actions[action]

        self.game.make_action(a)

        state = self.game.get_state()

        if state is not None:
            state = self.game.get_state().screen_buffer
        else:
            state = numpy.zeros(self.observation_shape, dtype=numpy.uint8)

        reward = self._shape_reward()

        done      = self.game.is_player_dead() or self.game.is_episode_finished()
        
        truncated = False  
      
        info  = self._make_info()

        return state, reward, done, truncated, info

    def close(self):
        self.game.close()

    def _shape_reward(self):
        self.frags_prev     = self.frags
        self.damage_prev    = self.damage
        self.health_prev    = self.health

        self.frags  = self.game.get_game_variable(vizdoom.vizdoom.GameVariable.FRAGCOUNT)
        self.damage = self.game.get_game_variable(vizdoom.vizdoom.GameVariable.DAMAGECOUNT)
        self.health = self.game.get_game_variable(vizdoom.vizdoom.GameVariable.HEALTH)
        self.health = max(0.0, self.health) 

        dfrags  = self.frags  - self.frags_prev
        ddamage = self.damage - self.damage_prev
        dhealth = self.health - self.health_prev


        x, y = self._get_player_position()

        dx = self.prev_x - x
        dy = self.prev_y - y
        self.prev_x = x
        self.prev_y = y  

        distance = numpy.sqrt(dx ** 2 + dy ** 2)

        if distance > 3.0:
            distance_reward = 1.0
        else:
            distance_reward = -1.0


        reward = 0.0
        reward+= 1.0*dfrags 
        reward+= 0.01*ddamage
        reward+= 0.01*dhealth
        reward+= 0.0005*distance_reward
        

        if self.game.is_player_dead() or self.game.is_episode_finished():
            reward-= 1.0

        reward = numpy.clip(reward, -10.0, 10.0)

        if dfrags > 0:
            self.episode_frags+= dfrags

        if ddamage > 0:
            self.episode_damage+= 1

        return reward
    
    def _make_info(self):
        info = {}
        info["frags"]  = self.episode_frags
        info["damage"] = self.episode_damage

        return info
    
    def _get_player_position(self):
        x, y = self.game.get_game_variable(vizdoom.vizdoom.GameVariable.POSITION_X), self.game.get_game_variable(vizdoom.vizdoom.GameVariable.POSITION_Y)
        return x, y

class BasicWrapper(gym.Wrapper ):
    def __init__(self, env, action_repeat, max_steps, env_name):
        super(BasicWrapper, self).__init__(env)

        self.env            = env
        self.action_repeat  = action_repeat
        self.max_steps      = max_steps
        self.steps          = 0
        self.env_name       = str(env_name)

    def step(self, action):
        reward_sum = 0
        for i in range(self.action_repeat):
            self.steps+= 1
            observation, reward, done, truncated, info = self.env.step(action)
            reward_sum+= reward
            if done:
                break
        
        if self.steps >= self.max_steps:
            done = True

        return observation, reward_sum, done, truncated, info
    
    def reset(self, seed = None, options = None):
        observation, info = self.env.reset()

        self.steps = 0

        return observation, info
    

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
    #print("maps count    = ", len(deatch_math_maps))
    #print("selecting map = ", map_id)

    env = CreateDoomGame(scenario_name, map_name = deatch_math_maps[map_id], render = render, n_bots=7)

    env = BasicWrapper(env, 4, 4500, env)
    env = StateWrapper(env, height, width, frame_stacking, False)

    return env


def DoomWrapperRender(env, height = 64, width = 128, frame_stacking = 2):
    return DoomWrapper(env, height, width, frame_stacking, True)
    