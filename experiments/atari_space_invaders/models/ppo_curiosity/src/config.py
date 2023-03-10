class Config(): 
    def __init__(self):
        self.gamma_ext              = 0.995
        self.gamma_int              = 0.95

        self.ext_adv_coeff          = 1.0
        self.int_adv_coeff          = 0.25
        self.normalize_motivation   = False

        self.entropy_beta           = 0.001
        self.eps_clip               = 0.2

        self.steps                  = 128
        self.batch_size             = 16
        
        self.training_epochs        = 4
        self.actors                 = 16
        
        self.learning_rate_ppo      = 0.00025
        self.learning_rate_forward  = 0.0001
        
