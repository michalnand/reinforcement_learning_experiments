import torch
import copy

class ModelBaseFeatures(torch.nn.Module):
    def __init__(self, input_shape):
        super(ModelBaseFeatures, self).__init__()

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
        )
        
        for i in range(len(self.model)):
            if hasattr(self.model[i], "weight"):
                torch.nn.init.orthogonal_(self.model[i].weight, 2**0.5)
                torch.nn.init.zeros_(self.model[i].bias)


    def forward(self, state):
        x = state[:,0,:,:].unsqueeze(1)
        return self.model(x) 
    


class ModelBasePredictor(torch.nn.Module):
    def __init__(self, features_count):
        super(ModelBasePredictor, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.ELU(),
            torch.nn.Linear(features_count, features_count),
            torch.nn.ELU(),
            torch.nn.Linear(features_count, features_count)
        )

        for i in range(len(self.model)):
            if hasattr(self.model[i], "weight"):
                torch.nn.init.orthogonal_(self.model[i].weight, 2**0.5)
                torch.nn.init.zeros_(self.model[i].bias)


    def forward(self, x):
        return self.model(x)

 

class Model(torch.nn.Module):
    def __init__(self, input_shape):
        super(Model, self).__init__()

        self.features_a = ModelBaseFeatures(input_shape)
        self.features_b = ModelBaseFeatures(input_shape)
        
        self.predictor_a = ModelBasePredictor(512)
        self.predictor_b = ModelBasePredictor(512)

        self.projector_a = torch.nn.Linear(512, 512)
        self.projector_b = torch.nn.Linear(512, 512)

        print("model_im")
        print(self.features_a) 
        print(self.predictor_a)
        print(self.projector_a)
        print("\n\n")

    
    
    def forward_a(self, state): 
        x = state[:,0,:,:].unsqueeze(1)
        return self.features_a(x)
    
    def forward_b(self, state): 
        x = state[:,0,:,:].unsqueeze(1)
        return self.features_b(x)
    
    def forward_predictor_a(self, z): 
        return self.predictor_a(z)
    
    def forward_predictor_b(self, z): 
        return self.predictor_b(z)
    
    def forward_projector_a(self, z): 
        return self.projector_a(z) 
    
    def forward_projector_b(self, z): 
        return self.projector_b(z)
    