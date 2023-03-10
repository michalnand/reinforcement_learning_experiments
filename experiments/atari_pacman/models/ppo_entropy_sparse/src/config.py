class Config(): 
    def __init__(self):
        self.gamma_ext              = 0.998
        self.gamma_int              = 0.99

        self.ext_adv_coeff                      = 1.0
        self.int_global_novelty_adv_coeff       = 0.25
        self.int_episodic_novelty_adv_coeff     = 0.25


        self.entropy_beta           = 0.001 
        self.eps_clip               = 0.2

        self.steps                  = 128
        self.batch_size             = 16
        
        self.training_epochs        = 4
        self.actors                 = 16
        
        self.learning_rate_ppo          = 0.00025
        self.learning_rate_forward      = 0.0001
        self.learning_rate_embeddings   = 0.0001
        
        self.episodic_novelty_memory_size       = 64
 
