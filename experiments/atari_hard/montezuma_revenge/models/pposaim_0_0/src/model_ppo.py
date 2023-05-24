import torch
import torch.nn as nn

class ResidualBlock(torch.nn.Module):
    def __init__(self, features_count):
        super(ResidualBlock, self).__init__()

        self.conv0 = torch.nn.Conv2d(features_count, features_count, kernel_size=3, stride=1, padding=1)
        self.act0  = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(features_count, features_count, kernel_size=3, stride=1, padding=1)
        self.act1  = torch.nn.ReLU()

        torch.nn.init.orthogonal_(self.conv0.weight, 2**0.5)
        torch.nn.init.zeros_(self.conv0.bias)

        torch.nn.init.orthogonal_(self.conv1.weight, 2**0.5)
        torch.nn.init.zeros_(self.conv1.bias)

    def forward(self, x):
        y = self.conv0(x)
        y = self.act0(y)
        y = self.conv1(y)
        y = y + x
        y = self.act1(y)

        return y
    

class ImpalaBlock(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(ImpalaBlock, self).__init__()

        self.conv0  = nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1)
        self.act0   = nn.ReLU()
        self.res0   = ResidualBlock(out_features)
        self.res1   = ResidualBlock(out_features)

        torch.nn.init.orthogonal_(self.conv0.weight, 2**0.5)
        torch.nn.init.zeros_(self.conv0.bias)

    def forward(self, x):
        y = self.conv0(x)
        y = self.act0(y)
        y = self.res0(y)
        y = self.res1(y)

        return y

class ModelFC(torch.nn.Module):
    def __init__(self, inputs_count, hidden_count, outputs_count, w0_init = 0.01, w1_init = 0.01):
        super(ModelFC, self).__init__()

        self.lin0 = torch.nn.Linear(inputs_count, hidden_count)
        self.act0 = torch.nn.ReLU()
        self.lin1 = torch.nn.Linear(hidden_count, outputs_count)

        torch.nn.init.orthogonal_(self.lin0.weight, w0_init)
        torch.nn.init.zeros_(self.lin0.bias)

        torch.nn.init.orthogonal_(self.lin1.weight, w1_init)
        torch.nn.init.zeros_(self.lin1.bias)

     
    def forward(self, x):
        y = self.lin0(x)
        y = self.act0(y)
        y = self.lin1(y)
        return y
    


class ModelCritic(torch.nn.Module):
    def __init__(self, inputs_count, hidden_count, w0_init = 0.01, w1_init = 0.01):
        super(ModelCritic, self).__init__()

        self.lin0 = torch.nn.Linear(inputs_count, hidden_count)
        self.act0 = torch.nn.ReLU()

        self.out_ext = torch.nn.Linear(hidden_count, 1)
        self.out_int = torch.nn.Linear(hidden_count, 1)


        torch.nn.init.orthogonal_(self.lin0.weight, w0_init)
        torch.nn.init.zeros_(self.lin0.bias)

        torch.nn.init.orthogonal_(self.out_ext.weight, w1_init)
        torch.nn.init.zeros_(self.out_ext.bias) 

        torch.nn.init.orthogonal_(self.out_int.weight, w1_init)
        torch.nn.init.zeros_(self.out_int.bias)

     
    def forward(self, x):
        y = self.lin0(x)
        y = self.act0(y)
        
        y_ext = self.out_ext(y)
        y_int = self.out_int(y)
        
        return y_ext, y_int
     
class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            
            nn.Flatten(),
            nn.Linear(features_count, hidden_count),
            nn.ReLU()
        ]     

        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_features[i].weight, 2**0.5)
                torch.nn.init.zeros_(self.layers_features[i].bias)

        self.model_features         = nn.Sequential(*self.layers_features)
 
        self.model_policy           = ModelFC(hidden_count, hidden_count, outputs_count, 0.01, 0.01)
        self.model_critic           = ModelCritic(hidden_count, hidden_count, 0.1, 0.01)
        self.model_self_supervised  = ModelFC(hidden_count, hidden_count, hidden_count, 0.01, 0.01)
        self.model_aux              = ModelFC(hidden_count, hidden_count, 3, 0.01, 0.01)

 
        self.model_features.to(self.device) 
        self.model_policy.to(self.device) 
        self.model_critic.to(self.device) 
        self.model_self_supervised.to(self.device) 
        self.model_aux.to(self.device)

        print("model_ppo")
        print(self.model_features)
        print(self.model_policy)
        print(self.model_critic)
        print(self.model_self_supervised)
        print(self.model_aux)
        print("\n\n")

    def forward(self, state):
        features                = self.model_features(state)
        policy                  = self.model_policy(features)
        value_ext, value_int    = self.model_critic(features)

        return policy, value_ext, value_int

    def forward_features(self, state):
        features    = self.model_features(state)
        return self.model_self_supervised(features)
    
    def forward_aux(self, sa, sb):
        za    = self.model_features(sa)
        zb    = self.model_features(sb)
        
        z     = za - zb

        return self.model_aux(z)
       
    def save(self, path):
        print("saving ", path)

        torch.save(self.model_features.state_dict(), path + "model_features.pt")
        torch.save(self.model_policy.state_dict(), path + "model_policy.pt")
        torch.save(self.model_critic.state_dict(), path + "model_critic.pt")
        torch.save(self.model_self_supervised.state_dict(), path + "model_self_supervised.pt")
        torch.save(self.model_aux.state_dict(), path + "model_aux.pt")

    def load(self, path):
        print("loading ", path) 

        self.model_features.load_state_dict(torch.load(path + "model_features.pt", map_location = self.device))
        self.model_policy.load_state_dict(torch.load(path + "model_policy.pt", map_location = self.device))
        self.model_critic.load_state_dict(torch.load(path + "model_critic.pt", map_location = self.device))
        self.model_self_supervised.load_state_dict(torch.load(path + "model_self_supervised.pt", map_location = self.device))
        self.model_aux.load_state_dict(torch.load(path + "model_aux.pt", map_location = self.device))

        self.model_features.eval()  
        self.model_policy.eval() 
        self.model_critic.eval()
        self.model_self_supervised.eval()
        self.model_aux.eval()


if __name__ == "__main__":
    state_shape     = (4, 96, 96)
    actions_count   = 18
    batch_size      = 32

    model           = Model(state_shape, actions_count)

    state           = torch.randn((batch_size, ) + state_shape)
    
    policy, ext_value = model(state)

    print("shape = ", policy.shape, ext_value.shape)
