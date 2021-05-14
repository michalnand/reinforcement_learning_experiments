import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, input_shape):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        fc_size = 64*(input_shape[1]//8) * (input_shape[2]//8)

        self.layers = [ 
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(fc_size, 256),
            nn.ReLU()
        ]

        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.orthogonal_(self.layers[i].weight, 2**0.5)
                torch.nn.init.zeros_(self.layers[i].bias)
                 
        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print("model_embeddings")
        print(self.model)
        print("\n\n")

    def forward(self, x):
        #x shape is (2, batch, channel, height, width)

        #select only 1st channel
        xa = x[0,:,0,:,:].unsqueeze(1)
        xb = x[1,:,0,:,:].unsqueeze(1)

        fa = self.model(xa)
        fb = self.model(xb)
        
        #euclidean metrics
        dist = ((fa - fb)**2).mean(dim=1)
        
        return dist

    def eval(self, state_t):
        return self.model(state_t)
       
    def save(self, path):
        print("saving ", path)
        torch.save(self.model.state_dict(),    path + "model_embeddings.pt")


    def load(self, path):
        print("loading ", path) 
        self.model.load_state_dict(torch.load(path + "model_embeddings.pt", map_location = self.device))
        self.model.eval() 


if __name__ == "__main__":
    state = torch.randn((2, 32, 4, 96, 96))

    model = Model((4, 96, 96))

    y = model.forward(state)

    print(state.shape, y.shape)
    print(y)
       
