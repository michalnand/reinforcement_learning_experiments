import torch
import torch.nn as nn


     
class Model(torch.nn.Module):

    def __init__(self, input_shape, out_features_count, aux_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_channels  = input_shape[0]
        input_height    = input_shape[1]
        input_width     = input_shape[2]    

        features_count  = 32*(input_height//8)*(input_width//8)
        hidden_count    = 256   
    
        self.layers_features = [ 
            torch.nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),

            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),

            torch.nn.Flatten(),

            torch.nn.Linear(features_count, hidden_count),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_count, out_features_count)
        ]     

        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_features[i].weight, 0.1)
                torch.nn.init.zeros_(self.layers_features[i].bias)


        self.layers_aux = [ 
            torch.nn.Linear(out_features_count, hidden_count),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_count, aux_count)
        ]
 
        for i in range(len(self.layers_aux)):
            if hasattr(self.layers_aux[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_aux[i].weight, 0.01)
                torch.nn.init.zeros_(self.layers_aux[i].bias)

        self.model_features         = nn.Sequential(*self.layers_features)
        self.model_aux              = nn.Sequential(*self.layers_aux)

        self.model_features.to(self.device) 
        self.model_aux.to(self.device) 
        
        print("model_im")
        print(self.model_features)
        print(self.model_aux)
        print("\n\n")

    def forward(self, state):
        y = self.model_features(state)       
        return y

    def forward_aux(self, sa, sb):
        za    = self.model_features(sa)
        zb    = self.model_features(sb)
        
        z     = za - zb

        return self.model_aux(z)
       
    def save(self, path):
        print("saving ", path)
        torch.save(self.model_features.state_dict(), path + "model_im_features.pt")
        torch.save(self.model_aux.state_dict(), path + "model_im_aux.pt")
        
    def load(self, path):
        print("loading ", path) 

        self.model_features.load_state_dict(torch.load(path + "model_im_features.pt", map_location = self.device))
        self.model_aux.load_state_dict(torch.load(path + "model_im_aux.pt", map_location = self.device))

        self.model_features.eval()  
        self.model_aux.eval()  
       
