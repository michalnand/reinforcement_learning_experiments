class Config(): 
    def __init__(self):
        #generic RL parameters
        self.envs_count         = 128
          
        self.gamma_ext          = 0.998
        self.gamma_int          = 0.99

        #learning rates
        self.learning_rate_ppo          = 0.0001 
        self.learning_rate_im           = 0.0001
   
        
        #reward scaling
        self.ext_adv_coeff          = 2.0
        self.int_adv_coeff          = 1.0
        self.reward_int_coeff       = 0.5
        self.reward_int_dif_coeff   = 0.5
        self.mi_loss_coeff          = 1.0


        #ppo params
        self.entropy_beta       = 0.001
        self.eps_clip           = 0.1
    
        self.steps              = 128
        self.batch_size         = 4
        self.training_epochs    = 4
        

        #internal motivation params    
        self.augmentations                  = ["pixelate", "random_tiles", "noise"]
        self.augmentations_probs            = 0.5
        
        self.similar_states_distance    = 4
        self.mode                       = "symmetric"
      
        #special params
        self.rnn_policy         = False
        self.state_normalise    = True

        
