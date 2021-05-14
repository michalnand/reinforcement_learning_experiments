import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, input_shape):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        fc_size = (input_shape[1]//16) * (input_shape[2]//16)
        self.layers = [
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            nn.Flatten(),  

            nn.Linear(64*fc_size, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 512)
        ]

        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers[i].weight)
                torch.nn.init.zeros_(self.layers[i].bias)
                
        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print("model_forward")
        print(self.model)
        print("\n\n")

    def forward(self, state): 
        return self.model(state)

    def save(self, path):
        torch.save(self.model.state_dict(), path + "model_forward.pt")
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path + "model_forward.pt", map_location = self.device))
        self.model.eval() 
