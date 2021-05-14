import torch
import torch.nn as nn

class ModelCritic(torch.nn.Module):
    def __init__(self, inputs_count, hidden_count):
        super(ModelCritic, self).__init__()

        self.hidden     = torch.nn.Linear(inputs_count, hidden_count)
        self.act        = torch.nn.ReLU() 

        self.ext_value              = torch.nn.Linear(hidden_count, 1)
        self.int_curiosity_value    = torch.nn.Linear(hidden_count, 1)
        self.int_entropy_value      = torch.nn.Linear(hidden_count, 1)

        torch.nn.init.orthogonal_(self.hidden.weight, 0.01)
        torch.nn.init.zeros_(self.hidden.bias)
        
        torch.nn.init.orthogonal_(self.ext_value.weight, 0.01)
        torch.nn.init.zeros_(self.ext_value.bias)

        torch.nn.init.orthogonal_(self.int_curiosity_value.weight, 0.01)
        torch.nn.init.zeros_(self.int_curiosity_value.bias)

        torch.nn.init.orthogonal_(self.int_entropy_value.weight, 0.01)
        torch.nn.init.zeros_(self.int_entropy_value.bias)

    def forward(self, x):
        y = self.act(self.hidden(x))

        return self.ext_value(y), self.int_curiosity_value(y), self.int_entropy_value(y)


class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_shape    = input_shape
        self.outputs_count  = outputs_count
        
        input_channels  = self.input_shape[0]
        input_height    = self.input_shape[1]
        input_width     = self.input_shape[2]    

        fc_inputs_count = 64*(input_width//8)*(input_height//8)
  
        self.layers_features = [ 
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Flatten(),

            nn.Linear(fc_inputs_count, 512),
            nn.ReLU()
        ]  

        self.layers_policy = [
            nn.Linear(512, 512),
            nn.ReLU(),                                          
            nn.Linear(512, outputs_count)
        ]
 
        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_features[i].weight, 2**0.5)
                torch.nn.init.zeros_(self.layers_features[i].bias)

        for i in range(len(self.layers_policy)):
            if hasattr(self.layers_policy[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_policy[i].weight, 0.01)
                torch.nn.init.zeros_(self.layers_policy[i].bias)

        self.model_features = nn.Sequential(*self.layers_features)
        self.model_features.to(self.device)

        self.model_critic = ModelCritic(512)
        self.model_critic.to(self.device)
        
        self.model_policy = nn.Sequential(*self.layers_policy)
        self.model_policy.to(self.device)

        print("model_ppo")
        print(self.model_features)
        print(self.model_critic)
        print(self.model_policy)
        print("\n\n")


    def forward(self, state):
        features                                            = self.model_features(state)

        ext_value, int_curiosity_value, int_entropy_value   = self.model_critic(features)
        policy                                              = self.model_policy(features)

        return policy, ext_value, int_curiosity_value, int_entropy_value

    def save(self, path):
        print("saving ", path)

        torch.save(self.model_features.state_dict(), path + "model_features.pt")
        torch.save(self.model_critic.state_dict(), path + "model_critic.pt")
        torch.save(self.model_policy.state_dict(), path + "model_policy.pt")

    def load(self, path):
        print("loading ", path) 

        self.model_features.load_state_dict(torch.load(path + "model_features.pt", map_location = self.device))
        self.model_critic.load_state_dict(torch.load(path + "model_critic.pt", map_location = self.device))
        self.model_policy.load_state_dict(torch.load(path + "model_policy.pt", map_location = self.device))
        
        self.model_features.eval()  
        self.model_critic.eval()
        self.model_policy.eval() 
