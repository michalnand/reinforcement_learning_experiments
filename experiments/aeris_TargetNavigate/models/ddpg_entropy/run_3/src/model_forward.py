import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, kernels_count = 32, hidden_count = 256):
        super(Model, self).__init__()

        self.device = "cpu"

        self.channels   = input_shape[0]
        self.width      = input_shape[1]

        fc_count        = kernels_count*self.width//4

        self.layers = [ 
            nn.Conv1d(self.channels + outputs_count, kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(fc_count, hidden_count),
            nn.ReLU(),            
            nn.Linear(hidden_count, hidden_count//2)           
        ] 

        torch.nn.init.orthogonal_(self.layers[0].weight, 2.0**0.5)
        torch.nn.init.orthogonal_(self.layers[3].weight, 2.0**0.5)
        torch.nn.init.orthogonal_(self.layers[5].weight, 2.0**0.5)
 
        self.model = nn.Sequential(*self.layers) 
        self.model.to(self.device)

        print("model_forward")
        print(self.model)
        print("\n\n")
       

    def forward(self, state, action):
        a_  = action.unsqueeze(2).repeat(1, 1, state.shape[2])
        x   = torch.cat([state, a_], dim = 1) 
      
        return self.model(x)

    def save(self, path):
        print("saving to ", path)
        torch.save(self.model.state_dict(), path + "model_forward.pt")

    def load(self, path):       
        print("loading from ", path)
        self.model.load_state_dict(torch.load(path + "model_forward.pt", map_location = self.device))
        self.model.eval()  



if __name__ == "__main__":
    batch_size      = 1
    input_shape     = (6, 32)
    outputs_count   = 5

    model = Model(input_shape, outputs_count)

    state   = torch.randn((batch_size, ) + input_shape)
    action  = torch.randn((batch_size, outputs_count))

    y = model.forward(state, action)

    print(y.shape)
