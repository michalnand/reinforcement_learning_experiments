class Config(): 
    def __init__(self):
        self.gamma_ext              = 0.998
        self.gamma_int              = 0.99

        #advanteges weighting
        self.ext_adv_coeff          = 2.0
        self.int_adv_coeff          = 1.0
        self.int_reward_coeff       = 1.0
        self.aux_loss_coeff         = 0.0

        self.entropy_beta           = 0.001
        self.eps_clip               = 0.1
        self.beta_var               = None
        self.state_normalise        = True

        self.steps                  = 128
        self.batch_size             = 4
        
        self.training_epochs        = 4
        self.envs_count             = 128
  
        #learning rates
        self.learning_rate_ppo          = 0.0001 
        self.learning_rate_cnd          = 0.0001
        self.learning_rate_cnd_target   = 0.0001

        #self supervised learning loss
        self.target_regularization_loss = "vicreg"
        self.target_aux_loss            = None
        
        #used augmentations
        self.augmentations              = ["pixelate", "random_tiles", "noise"]
        self.augmentations_probs        = 0.5