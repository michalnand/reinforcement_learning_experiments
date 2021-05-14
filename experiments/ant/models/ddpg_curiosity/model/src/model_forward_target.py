import torch
import torch.nn as nn


class Model(torch.nn.Module):
    def __init__(self, input_shape, hidden_count = 128):
        super(Model, self).__init__()

        self.device = "cpu"
        
        self.layers = [ 
            nn.Linear(input_shape[0], hidden_count),
            nn.ReLU(),            
            nn.Linear(hidden_count, hidden_count//4)
        ]

        torch.nn.init.orthogonal_(self.layers[0].weight, 2**0.5)
        torch.nn.init.orthogonal_(self.layers[2].weight, 2**0.5)

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print("model_forward_target")
        print(self.model)
        print("\n\n")
       
    def forward(self, state):
        return self.model(state)

    def save(self, path):
        print("saving to ", path)
        torch.save(self.model.state_dict(), path + "model_forward_target.pt")

    def load(self, path):       
        print("loading from ", path)
        self.model.load_state_dict(torch.load(path + "model_forward_target.pt", map_location = self.device))
        self.model.eval()  
    
