import torch

class ModelFC(torch.nn.Module):
    def __init__(self, inputs_count, hidden_count, outputs_count):
        super(ModelFC, self).__init__()

        self.act0 = torch.nn.ReLU()
        self.lin0 = torch.nn.Linear(inputs_count, hidden_count)
        self.act1 = torch.nn.ReLU()
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

        self.act0       = torch.nn.ReLU()
        self.hidden     = torch.nn.Linear(inputs_count, hidden_count)
        
        self.act1       = torch.nn.ReLU()

        self.value  = torch.nn.Linear(hidden_count, 1)

        torch.nn.init.orthogonal_(self.hidden.weight, 0.1)
        torch.nn.init.zeros_(self.hidden.bias)

        torch.nn.init.orthogonal_(self.value.weight, 0.01)
        torch.nn.init.zeros_(self.value.bias)

     
    def forward(self, x): 
        y = self.act0(x)
        y = self.hidden(y)
        y = self.act1(y)
        
        return self.value(y)

 
class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.input_shape    = input_shape
        self.outputs_count  = outputs_count
        
        input_channels  = self.input_shape[0]
        input_height    = self.input_shape[1]
        input_width     = self.input_shape[2]    

        features_count  = 64*(input_height//8)*(input_width//8)
        hidden_count    = 512   
        
        self.layers_features = [   
            torch.nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),

            torch.nn.Flatten(),
            torch.nn.Linear(features_count, hidden_count),
        ]     

        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_features[i].weight, 2**0.5)
                torch.nn.init.zeros_(self.layers_features[i].bias)

        self.model_features         = torch.nn.Sequential(*self.layers_features)
        self.model_policy           = ModelFC(hidden_count, hidden_count, outputs_count)
        self.model_critic           = ModelCritic(hidden_count, hidden_count)


        print("model_ppo")
        print(self.model_features)
        print(self.model_policy)
        print(self.model_critic)
        print("\n\n")

    def forward(self, state):
        features    = self.model_features(state)
        policy      = self.model_policy(features)
        value       = self.model_critic(features)

        return policy, value


    
       
if __name__ == "__main__":
    state_shape     = (4, 96, 96)
    actions_count   = 18
    batch_size      = 32

    model           = Model(state_shape, actions_count)

    state           = torch.randn((batch_size, ) + state_shape)
    
    policy, value = model(state)

    print("shape = ", policy.shape, value.shape)
