import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, input_shape):
        super(Model, self).__init__()

        self.device = "cpu"

        self.layers = [
            nn.Linear(input_shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        ] 

        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.orthogonal_(self.layers[i].weight, 2.0**0.5)
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

if __name__ == "__main__":
    batch_size = 8

    channels = 3
    height   = 96
    width    = 96

    actions_count = 9


    state           = torch.rand((batch_size, channels, height, width))
    action          = torch.rand((batch_size, actions_count))

    model = Model((channels, height, width), actions_count)

    state_predicted = model.forward(state, action)

    print(state_predicted.shape)


