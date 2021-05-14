class Config(): 
    def __init__(self):
        self.gamma_ext              = 0.998
        self.gamma_int              = 0.99

        self.ext_adv_coeff          = 1.0
        self.int_curiosity_adv_coeff= 0.5
        self.int_entropy_adv_coeff  = 0.001
 
        self.entropy_beta           = 0.001
        self.eps_clip               = 0.1

        self.steps                  = 128
        self.batch_size             = 4
        
        self.training_epochs        = 4
        self.actors                 = 128
        
        self.learning_rate_ppo          = 0.0001
        self.learning_rate_forward      = 0.0001
        self.learning_rate_embeddings   = 0.0001
        
        self.episodic_memory_size       = 512
 
 