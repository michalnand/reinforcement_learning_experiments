import torch
import torch.nn as nn


class Model(torch.nn.Module):
    def __init__(self, input_shape, actions_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        fc_size = (input_shape[1]//8) * (input_shape[2]//8)

        self.layers = [
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
          
            nn.Flatten(),   

            nn.Linear(64*fc_size, 512)
        ]

        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.orthogonal_(self.layers[i].weight, 2**0.5)
                torch.nn.init.zeros_(self.layers[i].bias)

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        self.layers_aux = [
            nn.LayerNorm((512, )),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(), 
            nn.Linear(256, 3)
        ]

        for i in range(1, len(self.layers_aux)):
            if hasattr(self.layers_aux[i], "weight"):
                torch.nn.init.orthogonal_(self.layers_aux[i].weight, 0.01)
                torch.nn.init.zeros_(self.layers_aux[i].bias)

        self.model_aux = nn.Sequential(*self.layers_aux)
        self.model_aux.to(self.device)

        print("model_cnd_target")
        print(self.model)
        print(self.model_aux)
        print("\n\n")

    def forward(self, state): 
        x = state[:,0,:,:].unsqueeze(1)
        return self.model(x)
    
    def forward_aux(self, state_a, state_b):
        xa = state_a[:,0,:,:].unsqueeze(1)
        xb = state_b[:,0,:,:].unsqueeze(1)

        za = self.model(xa)
        zb = self.model(xb)

        z   = za - zb

        return self.model_aux(z)

    def save(self, path):
        torch.save(self.model.state_dict(), path + "model_cnd_target.pt")
        torch.save(self.model_aux.state_dict(), path + "model_cnd_target_aux.pt")

    def load(self, path):
        self.model.load_state_dict(torch.load(path + "model_cnd_target.pt", map_location = self.device))
        self.model_aux.load_state_dict(torch.load(path + "model_cnd_target_aux.pt", map_location = self.device))
