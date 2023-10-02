import torch
import torch.nn as nn


class Model(torch.nn.Module):
    def __init__(self, input_shape):
        super(Model, self).__init__()

        fc_size = (input_shape[1]//8) * (input_shape[2]//8)

        self.layers = [
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
          
            nn.Flatten(),   

            nn.Linear(64*fc_size, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),

            nn.Linear(512, 512)
        ]

        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.orthogonal_(self.layers[i].weight, 2**0.5)
                torch.nn.init.zeros_(self.layers[i].bias)

        self.model = nn.Sequential(*self.layers)

        print("model_snd")
        print(self.model) 
        print("\n\n")

    def forward(self, state): 
        x = state[:,0:3,:,:]
        return self.model(x)

   

    
if __name__ == "__main__":
    state_shape     = (4*3, 96, 96)
    batch_size      = 32

    model           = Model(state_shape)

    state           = torch.randn((batch_size, ) + state_shape)
    
    z = model(state)

    print("shape = ", z.shape)
