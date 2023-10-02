import torch
import torch.nn as nn


class Residual(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Residual, self).__init__()


        self.conv0 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.act0  = torch.nn.ReLU()

        self.conv1 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.act1  = torch.nn.ReLU()


        torch.nn.init.orthogonal_(self.conv0.weight, 2**0.5)
        torch.nn.init.zeros_(self.conv0.bias)

        torch.nn.init.orthogonal_(self.conv1.weight, 2**0.5)
        torch.nn.init.zeros_(self.conv1.bias)

        if in_channels != out_channels or stride != 1:
            self.conv_bp = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

            torch.nn.init.orthogonal_(self.conv_bp.weight, 2**0.5)
            torch.nn.init.zeros_(self.conv_bp.bias)
        else:
            self.conv_bp = None

       
     
    def forward(self, x):

        y = self.conv0(x)
        y = self.act0(y)
        y = self.conv1(y)

        if self.conv_bp is not None:
            y = y + self.conv_bp(x)
        else:
            y = y + x

        y = self.act1(y)
        return y

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
        
        input_channels  = self.input_shape[0]
        input_height    = self.input_shape[1]
        input_width     = self.input_shape[2]    

        features_count  = 128*(input_height//8)*(input_width//8)
        hidden_count    = 512   
        
        self.layers_features = [   
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            Residual(32, 64, 2),
            Residual(64, 128, 2),
            
            nn.Flatten(),
            nn.Linear(features_count, hidden_count),
        ]     

        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_features[i].weight, 2**0.5)
                torch.nn.init.zeros_(self.layers_features[i].bias)

        self.model_features         = nn.Sequential(*self.layers_features)
        self.model_policy           = ModelFC(hidden_count, hidden_count, outputs_count)
        self.model_critic           = ModelCritic(hidden_count, hidden_count)
        self.model_self_supervised  = ModelFC(hidden_count, hidden_count, hidden_count)


        print("model_ppo")
        print(self.model_features)
        print(self.model_policy)
        print(self.model_critic)
        print(self.model_self_supervised)
        print("\n\n")

    def forward(self, state):
        features                = self.model_features(state)
        policy                  = self.model_policy(features)
        ext_value, int_value    = self.model_critic(features)

        return policy, ext_value, int_value

    def forward_features(self, state):
        features = self.model_features(state)
        y        = self.model_self_supervised(features)

        return y

    
       
if __name__ == "__main__":
    state_shape     = (4, 96, 96)
    actions_count   = 18
    batch_size      = 32

    model           = Model(state_shape, actions_count)

    state           = torch.randn((batch_size, ) + state_shape)
    
    policy, ext_value, int_value = model(state)

    print("shape = ", policy.shape, ext_value.shape, int_value.shape)
