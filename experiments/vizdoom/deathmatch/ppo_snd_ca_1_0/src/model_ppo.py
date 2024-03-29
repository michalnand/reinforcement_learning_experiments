import torch

class ModelFC(torch.nn.Module):
    def __init__(self, inputs_count, hidden_count, outputs_count):
        super(ModelFC, self).__init__()

        self.act0 = torch.nn.Tanh()
        self.lin0 = torch.nn.Linear(inputs_count, hidden_count)
        self.act1 = torch.nn.Tanh()
        self.lin1 = torch.nn.Linear(hidden_count, outputs_count)

        torch.nn.init.orthogonal_(self.lin0.weight, 0.01)
        torch.nn.init.zeros_(self.lin0.bias)

        torch.nn.init.orthogonal_(self.lin1.weight, 0.01)
        torch.nn.init.zeros_(self.lin1.bias)

     
    def forward(self, x):
        y = self.act0(x)
        y = self.lin0(y)
        y = self.act0(y)
        y = self.lin1(y)

        return y
    
class ModelCritic(torch.nn.Module): 
    def __init__(self, inputs_count, hidden_count):
        super(ModelCritic, self).__init__()
 
        self.act0       = torch.nn.Tanh()
        self.hidden     = torch.nn.Linear(inputs_count, hidden_count)
        
        self.act1       = torch.nn.Tanh()

        self.ext_value  = torch.nn.Linear(hidden_count, 1)
        self.int_value  = torch.nn.Linear(hidden_count, 1)

        torch.nn.init.orthogonal_(self.hidden.weight, 0.1)
        torch.nn.init.zeros_(self.hidden.bias)

        torch.nn.init.orthogonal_(self.ext_value.weight, 0.01)
        torch.nn.init.zeros_(self.ext_value.bias)

        torch.nn.init.orthogonal_(self.int_value.weight, 0.01)
        torch.nn.init.zeros_(self.int_value.bias)

    def forward(self, x): 
        y = self.act0(x)
        y = self.hidden(y)
        y = self.act1(y)
        
        return self.ext_value(y), self.int_value(y)

 
class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.input_shape    = input_shape
        self.outputs_count  = outputs_count
        self.rnn_size       = 256
        
        #use RGB colored input
        self.in_ch      = 3
        input_height    = self.input_shape[1]
        input_width     = self.input_shape[2]    

        features_count  = 64*(input_height//8)*(input_width//8)
        hidden_count    = 512   
 
        
        self.layers_features = [    
            torch.nn.Conv2d(self.in_ch, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),

            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(), 

            torch.nn.Flatten(),
            torch.nn.Linear(features_count, hidden_count),
        ]     

        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_features[i].weight, 2**0.5)
                torch.nn.init.zeros_(self.layers_features[i].bias)

        self.model_rnn = torch.nn.GRUCell(hidden_count, self.rnn_size)     

        torch.nn.init.orthogonal_(self.model_rnn.weight_ih, 0.1)
        torch.nn.init.orthogonal_(self.model_rnn.weight_hh, 0.1)
        torch.nn.init.zeros_(self.model_rnn.bias_ih)  
        torch.nn.init.zeros_(self.model_rnn.bias_hh)

        self.model_features         = torch.nn.Sequential(*self.layers_features)
        self.model_policy           = ModelFC(hidden_count + self.rnn_size, hidden_count, outputs_count)
        self.model_critic           = ModelCritic(hidden_count + self.rnn_size, hidden_count)
        self.model_self_supervised  = ModelFC(hidden_count, hidden_count, hidden_count)


        print("model_ppo")
        print(self.model_features)
        print(self.model_rnn)
        print(self.model_policy)
        print(self.model_critic)
        print(self.model_self_supervised)
        print("\n\n")

    def forward(self, state, hidden_state):
        hidden_state_new = hidden_state.detach()
        
        #obtain features in reversed order - newest frame enters as last into RNN
        for i in reversed(range(state.shape[1]//self.in_ch)):
            ofs       = i*self.in_ch
            s         = state[:, ofs:ofs + self.in_ch, :, :]

            features  = self.model_features(s)

            hidden_state_new = self.model_rnn(features, hidden_state_new)

        z = torch.concatenate([features, hidden_state_new], dim=1)
    
        policy                  = self.model_policy(z)
        ext_value, int_value    = self.model_critic(z)

        return policy, ext_value, int_value, hidden_state_new

    def forward_features(self, state):
        features = self.model_features(state[:, 0:self.in_ch, :, :])
        y        = self.model_self_supervised(features)

        return y

    
       
if __name__ == "__main__":
    state_shape     = (6, 96, 96)
    actions_count   = 18
    batch_size      = 32

    model           = Model(state_shape, actions_count)

    state           = torch.randn((batch_size, ) + state_shape)
    hidden_state    = torch.randn((batch_size, model.rnn_size))
    
    policy, ext_value, int_value, hidden_new = model(state, hidden_state)

    print("shape = ", policy.shape, ext_value.shape, int_value.shape, hidden_new.shape)
