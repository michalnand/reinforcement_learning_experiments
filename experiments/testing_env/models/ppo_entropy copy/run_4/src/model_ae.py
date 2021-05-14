import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, input_shape):
        super(Model, self).__init__()

        self.device = "cpu"

        self.layers_encoder = [ 
            nn.Linear(input_shape[0], 64),
            nn.ELU(),
            nn.Linear(64, 8)
        ]  

        self.layers_decoder = [
            nn.Linear(8, 64),
            nn.ELU(),
            nn.Linear(64, input_shape[0])
        ]   
   
        for i in range(len(self.layers_encoder)):
            if hasattr(self.layers_encoder[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_encoder[i].weight, 2**0.5)
                self.layers_encoder[i].bias.data.zero_()

        for i in range(len(self.layers_decoder)):
            if hasattr(self.layers_decoder[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_decoder[i].weight, 2**0.5)
                self.layers_decoder[i].bias.data.zero_()
 
        self.model_encoder = nn.Sequential(*self.layers_encoder)
        self.model_encoder.to(self.device)

        self.model_decoder = nn.Sequential(*self.layers_decoder)
        self.model_decoder.to(self.device)

        self.model_encoder = nn.Sequential(*self.layers_encoder)
        self.model_encoder.to(self.device)

        print("model_autoencoder")
        print(self.model_encoder)
        print(self.model_decoder)
        print("\n\n")

    def forward(self, x, noise_enabled = True):
        z       = self.model_encoder(x)
        return self.model_decoder(z), z.view((z.shape[0], -1))

    def eval_features(self, x):
        z   = self.model_encoder(x)
        return z.view((z.shape[0], -1))
       
    def save(self, path):
        print("saving ", path)

        torch.save(self.model_encoder.state_dict(), path + "model_ae_encoder.pt")
        torch.save(self.model_decoder.state_dict(), path + "model_ae_decoder.pt")

    def load(self, path):
        print("loading ", path) 

        self.model_encoder.load_state_dict(torch.load(path + "model_ae_encoder.pt", map_location = self.device))
        self.model_decoder.load_state_dict(torch.load(path + "model_ae_decoder.pt", map_location = self.device))

        self.model_encoder.eval() 
        self.model_decoder.eval() 
       
