import torch
import torch.nn as nn

class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.device = "cpu"
        hidden_size = 64
        
        self.layers_features = [  
            nn.Linear(input_shape[0], hidden_size),
            nn.ReLU(),
        ]

        self.layers_ext_value = [
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),                       
            nn.Linear(hidden_size, 1)    
        ]

        self.layers_int_value = [
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),                       
            nn.Linear(hidden_size, 1)    
        ]  

        self.layers_policy = [
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),                      
            nn.Linear(hidden_size, outputs_count)
        ]
 
  
        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_features[i].weight)

        for i in range(len(self.layers_ext_value)):
            if hasattr(self.layers_ext_value[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_ext_value[i].weight)

        for i in range(len(self.layers_int_value)):
            if hasattr(self.layers_int_value[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_int_value[i].weight)

        for i in range(len(self.layers_policy)):
            if hasattr(self.layers_policy[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_policy[i].weight)


        self.model_features = nn.Sequential(*self.layers_features)
        self.model_features.to(self.device)

        self.model_ext_value = nn.Sequential(*self.layers_ext_value)
        self.model_ext_value.to(self.device)

        self.model_int_value = nn.Sequential(*self.layers_int_value)
        self.model_int_value.to(self.device)

        self.model_policy = nn.Sequential(*self.layers_policy)
        self.model_policy.to(self.device)

        print("model_ppo")
        print(self.model_features)
        print(self.model_ext_value)
        print(self.model_int_value)
        print(self.model_policy)
        print("\n\n")


    def forward(self, state):
        features        = self.model_features(state)

        ext_value       = self.model_ext_value(features)
        int_value       = self.model_int_value(features)
        policy          = self.model_policy(features)

        return policy, ext_value, int_value

    def save(self, path):
        print("saving ", path)

        torch.save(self.model_features.state_dict(), path + "model_features.pt")
        torch.save(self.model_ext_value.state_dict(), path + "model_ext_value.pt")
        torch.save(self.model_int_value.state_dict(), path + "model_int_value.pt")
        torch.save(self.model_policy.state_dict(), path + "model_policy.pt")

    def load(self, path):
        print("loading ", path) 

        self.model_features.load_state_dict(torch.load(path + "model_features.pt", map_location = self.device))
        self.model_ext_value.load_state_dict(torch.load(path + "model_ext_value.pt", map_location = self.device))
        self.model_int_value.load_state_dict(torch.load(path + "model_int_value.pt", map_location = self.device))
        self.model_policy.load_state_dict(torch.load(path + "model_policy.pt", map_location = self.device))
        
        self.model_features.eval() 
        self.model_ext_value.eval()
        self.model_int_value.eval() 
        self.model_policy.eval() 


    def get_activity_map(self, state):
 
        state_t     = torch.tensor(state, dtype=torch.float32).detach().to(self.device).unsqueeze(0)
        features    = self.model_features(state_t)
        features    = features.reshape((1, 128, 6, 6))

        upsample = nn.Upsample(size=(self.input_shape[1], self.input_shape[2]), mode='bicubic')

        features = upsample(features).sum(dim = 1)

        result = features[0].to("cpu").detach().numpy()

        k = 1.0/(result.max() - result.min())
        q = 1.0 - k*result.max()
        result = k*result + q
        
        return result
