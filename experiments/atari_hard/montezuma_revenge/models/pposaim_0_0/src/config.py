class Config(): 
    def __init__(self):
        self.gamma_ext              = 0.998
        self.gamma_int              = 0.99
        self.int_reward_coeff       = 0.5
        
        self.entropy_beta           = 0.001
        self.eps_clip               = 0.1

        self.steps                  = 128
        self.batch_size             = 4
        
        self.training_epochs        = 4
        self.envs_count             = 128 
        
        self.learning_rate_ppo          = 0.0001
        self.learning_rate_im           = 0.0001
  
        
        self.self_supervised_loss_coeff = 0.1
        self.self_aware_loss_coeff      = 0.1

        self.im_self_supervised_loss_coeff  = 1.0
        self.im_self_aware_loss_coeff       = 1.0


        self.self_supervised_loss       = "vicreg"
        self.self_aware_loss            = "constructor_loss"
        self.im_buffer_size             = 4500

        self.augmentations              = ["aug_inverse", "pixelate", "random_tiles", "noise"]
        self.augmentations_probs        = 0.5

        
