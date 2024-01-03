class Config(): 
    def __init__(self):
        self.envs_count         = 128

        self.gamma              = 0.998

        self.entropy_beta       = 0.001
        self.eps_clip           = 0.1 

        self.steps              = 128
        self.batch_size         = 4
        
        self.training_epochs    = 4
        
        self.learning_rate     = 0.0001
        
        self.rnn_policy        = True   

        self.self_supervised_loss       = None