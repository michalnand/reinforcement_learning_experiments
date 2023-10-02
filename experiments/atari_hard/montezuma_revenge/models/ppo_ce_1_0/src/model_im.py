import torch

class Model(torch.nn.Module):
    def __init__(self, input_shape):
        super(Model, self).__init__()

        fc_size = (input_shape[1]//8) * (input_shape[2]//8)


        self.model_target = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ELU(),

            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ELU(),

            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ELU(),

            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ELU(),
          
            torch.nn.Flatten(),   

            torch.nn.Linear(64*fc_size, 512)
        )


        self.model_predictor = torch.nn.Sequential(
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


        for i in range(len(self.model_target)):
            if hasattr(self.model_target[i], "weight"):
                torch.nn.init.orthogonal_(self.model_target[i].weight, 2**0.5)
                torch.nn.init.zeros_(self.model_target[i].bias)
            
        for i in range(len(self.model_predictor)):
            if hasattr(self.model_predictor[i], "weight"):
                torch.nn.init.orthogonal_(self.model_predictor[i].weight, 2**0.5)
                torch.nn.init.zeros_(self.model_predictor[i].bias)

      
        print("model_im")
        print(self.model_target) 
        print("\n\n")
        print(self.model_predictor)
        print("\n\n")

    def forward(self, state): 
        x = state[:,0,:,:].unsqueeze(1)

        z_target    = self.model_target(x)
        z_predictor = self.model_predictor(x)

        return z_target, z_predictor
    

    def forward_features(self, state):
        x   = state[:,0,:,:].unsqueeze(1)
        z = self.model_target(x)

        return z



if __name__ == "__main__":

    input_shape = (4, 96, 96)
    batch_size  = 5

    x = torch.randn((batch_size, ) + input_shape)

    model = Model(input_shape)

    z_target, z_predictor = model(x)

    print("z_target ", z_target.shape)
    print("z_predictor ", z_predictor.shape)
   