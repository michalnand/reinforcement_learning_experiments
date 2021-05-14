import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, hidden_count = 128):
        super(Model, self).__init__()

        self.device = "cpu"

        self.layers_features = [ 
            nn.Linear(input_shape[0] + outputs_count, hidden_count),
            nn.ReLU(),
            nn.Linear(hidden_count, hidden_count//2),
            nn.ReLU()
        ]

        self.layers_ext = [
            nn.Linear(hidden_count//2, 1)           
        ] 

        self.layers_int = [          
            nn.Linear(hidden_count//2, 1)           
        ] 

        torch.nn.init.xavier_uniform_(self.layers_features[0].weight)
        torch.nn.init.xavier_uniform_(self.layers_features[2].weight)
        
        torch.nn.init.uniform_(self.layers_ext[0].weight, -0.003, 0.003)
        torch.nn.init.uniform_(self.layers_int[0].weight, -0.003, 0.003)
 
        self.model_features = nn.Sequential(*self.layers_features) 
        self.model_ext      = nn.Sequential(*self.layers_ext) 
        self.model_int      = nn.Sequential(*self.layers_int) 
        
        self.model_features.to(self.device)
        self.model_ext.to(self.device)
        self.model_int.to(self.device)

        print("model_critic")
        print(self.model_features)
        print(self.model_ext)
        print(self.model_int)
        print("\n\n")
       

    def forward(self, state, action):
        x           = torch.cat([state, action], dim = 1)
        features    = self.model_features(x)

        return self.model_ext(features), self.model_int(features)

     
    def save(self, path):
        print("saving to ", path)
        torch.save(self.model_features.state_dict(), path + "model_critic_features.pt")
        torch.save(self.model_ext.state_dict(), path + "model_critic_ext.pt")
        torch.save(self.model_int.state_dict(), path + "model_critic_int.pt")

    def load(self, path):       
        print("loading from ", path)
        self.model_features.load_state_dict(torch.load(path + "model_critic_features.pt", map_location = self.device))
        self.model_ext.load_state_dict(torch.load(path + "model_critic_exmodel_ext.pt", map_location = self.device))
        self.model_int.load_state_dict(torch.load(path + "model_critic_int.pt", map_location = self.device))

        self.model_features.eval()  
        self.model_ext.eval()  
        self.model_int.eval()  
    
