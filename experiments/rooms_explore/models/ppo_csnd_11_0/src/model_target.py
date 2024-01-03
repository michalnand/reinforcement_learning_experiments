import torch



class Model(torch.nn.Module):
    def __init__(self, input_shape):
        super(Model, self).__init__()

        fc_size = (input_shape[1]//4) * (input_shape[2]//4)

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ELU(),

            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ELU(),

            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ELU(),

            torch.nn.Flatten(),   

            torch.nn.Linear(64*fc_size, 512)
        ) 

        self.model_causality = torch.nn.Linear(512, 1)

        for i in range(len(self.model)):
            if hasattr(self.model[i], "weight"):
                torch.nn.init.orthogonal_(self.model[i].weight, 2**0.5)
                torch.nn.init.zeros_(self.model[i].bias)

        torch.nn.init.orthogonal_(self.model_causality.weight, 0.1)
        torch.nn.init.zeros_(self.model_causality.bias)

        print("model_target")
        print(self.model)
        print(self.model_causality)
        print("\n\n")

    def forward(self, x):
        return self.model(x)
    
    #z.shape = (batch, features_count)
    #returns : (batch, 1)
    def forward_causality(self, za, zb): 
        dz = za - zb 
        return self.model_causality(dz)
    
