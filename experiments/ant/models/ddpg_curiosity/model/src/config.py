import RLAgents

class Config():

    def __init__(self):        
        self.gamma_ext              = 0.99
        self.gamma_int              = 0.95
        self.ext_coeff              = 1.0
        self.int_coeff              = 0.5
        
        self.critic_learning_rate   = 0.0002
        self.actor_learning_rate    = 0.0001
        self.forward_learning_rate  = 0.0001
        
        self.tau                    = 0.001
 
        self.batch_size             = 64
        self.update_frequency       = 4

        self.exploration            = RLAgents.lib_common.DecayLinear(1000000, 0.3, 0.05, 0.05)

        self.experience_replay_size = 200000
