import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        fc_size = (input_shape[1]//16) * (input_shape[2]//16)
        
        self.layers_features = [ 
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=2),
            nn.ELU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU()
        ] 
        
        self.layers_output = [
            nn.Linear(2*fc_size*64, 512),
            nn.ELU(),
            nn.Linear(512, outputs_count)
        ]   
   
        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_features[i].weight, 2**0.5)
                self.layers_features[i].bias.data.zero_()

        for i in range(len(self.layers_output)):
            if hasattr(self.layers_output[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_output[i].weight)
                self.layers_output[i].bias.data.zero_()
 
        self.model_features = nn.Sequential(*self.layers_features)
        self.model_features.to(self.device)

        self.model_output = nn.Sequential(*self.layers_output)
        self.model_output.to(self.device)

      
        print("model_embeddings")
        print(self.model_features)
        print(self.model_output)
        print("\n\n")

    def forward(self, state, state_next):
        z0   = self.model_features(state)
        z1   = self.model_features(state_next)

        z0 = z0.view((z0.shape[0], -1))
        z1 = z1.view((z1.shape[0], -1))

        z  = torch.cat([z0, z1], dim=1)

        return self.model_output(z)

    def eval_features(self, state):
        z = self.model_features(state)
        return z.view((z.shape[0], -1))
       
    def save(self, path):
        print("saving ", path)

        torch.save(self.model_features.state_dict(), path + "model_embeddings_encoder.pt")
        torch.save(self.model_output.state_dict(), path + "model_embeddings_decoder.pt")

    def load(self, path):
        print("loading ", path) 

        self.model_features.load_state_dict(torch.load(path + "model_embeddings_encoder.pt", map_location = self.device))
        self.model_output.load_state_dict(torch.load(path + "model_embeddings_decoder.pt", map_location = self.device))

        self.model_features.eval() 
        self.model_output.eval() 
       
