import torch

class Model(torch.nn.Module):
    def __init__(self, input_shape):
        super(Model, self).__init__()

        fc_size = (input_shape[1]//8) * (input_shape[2]//8)

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ELU(),

            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ELU(),

            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ELU(),

            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ELU(),
          
            torch.nn.Flatten(),   

            torch.nn.Linear(64*fc_size, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 512),
            torch.nn.ELU(),

            torch.nn.Linear(512, 512)
        )

        for i in range(len(self.model)):
            if hasattr(self.model[i], "weight"):
                torch.nn.init.orthogonal_(self.model[i].weight, 2**0.5)
                torch.nn.init.zeros_(self.model[i].bias)

        print("model_predictor")
        print(self.model)
        print("\n\n")

    def forward(self, x):
        x_tmp = x[:,0,:,:].unsqueeze(1)
        return self.model(x_tmp)
  

