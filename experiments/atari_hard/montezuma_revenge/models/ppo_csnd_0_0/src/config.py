class Config(): 
    def __init__(self):
        self.gamma_ext              = 0.998
        self.gamma_int              = 0.99

        #advanteges weighting
        self.ext_adv_coeff          = 2.0
        self.int_adv_coeff          = 1.0

        self.reward_int_a_coeff     = 0.5
        self.reward_int_b_coeff     = 0.0
        self.reward_int_dif_coeff   = 0.5
        self.causality_loss_coeff   = 0.0
        self.contextual_buffer_size = 256

        self.entropy_beta           = 0.001
        self.eps_clip               = 0.1
        self.rnn_policy             = False

        self.steps                  = 128
        self.batch_size             = 4
        
        self.training_epochs        = 4
        self.envs_count             = 128

   
        #learning rates
        self.learning_rate_ppo          = 0.0001 
        self.learning_rate_target       = 0.0001
        self.learning_rate_predictor    = 0.0001
        
        #self supervised learning loss
        self.ppo_self_supervised_loss       = "vicreg"
        self.target_self_supervised_loss    = "vicreg"

        self.similar_states_distance        = 4
        self.state_normalise                = True
        
        #used augmentations
        self.augmentations              = ["pixelate", "random_tiles", "noise"]
        self.augmentations_probs        = 0.5 