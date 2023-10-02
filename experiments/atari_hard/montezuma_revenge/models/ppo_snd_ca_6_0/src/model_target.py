import torch
import torch.nn as nn


class Model(torch.nn.Module):
    def __init__(self, input_shape, actions_count):
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

            nn.Linear(64*fc_size, 512)
        ]

        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.orthogonal_(self.layers[i].weight, 2**0.5)
                torch.nn.init.zeros_(self.layers[i].bias)

        self.model = nn.Sequential(*self.layers)

        self.model_aux = nn.Sequential(
            nn.LayerNorm((512, )),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(), 
            nn.Linear(256, 3)
        )

        for i in range(1, len(self.model_aux)):
            if hasattr(self.model_aux[i], "weight"):
                torch.nn.init.orthogonal_(self.model_aux[i].weight, 0.1)
                torch.nn.init.zeros_(self.model_aux[i].bias)

     
        print("model_snd_target")
        print(self.model)
        print(self.model_aux)
        print("\n\n")

    def forward(self, state): 
        x = state[:,0:3,:,:]
        return self.model(x)
    

    def forward_aux(self, state_a, state_b): 
        xa = state_a[:,0:3,:,:]
        xb = state_b[:,0:3,:,:]

        za = self.model(xa)
        zb = self.model(xb)

        return self.model_aux(za - zb)
    

     
if __name__ == "__main__":
    state_shape     = (4*3, 96, 96)
    actions_count   = 18
    batch_size      = 32

    model           = Model(state_shape, actions_count)

    state           = torch.randn((batch_size, ) + state_shape)
    
    z = model(state)

    print("shape = ", z.shape)
