import torch


class ModelCausality(torch.nn.Module):
    def __init__(self, in_ch, hidden_ch):
        super(ModelCausality, self).__init__()

        self.conv0 = torch.nn.Conv2d(in_ch, hidden_ch, kernel_size=1, stride=1, padding=0)
        self.act0  = torch.nn.LeakyReLU()
        
        self.conv1 = torch.nn.Conv2d(hidden_ch, hidden_ch, kernel_size=1, stride=1, padding=0)
        self.act1  = torch.nn.LeakyReLU()

        self.conv2 = torch.nn.Conv2d(hidden_ch, 1, kernel_size=1, stride=1, padding=0)
        self.act2  = torch.nn.Sigmoid()

       
        torch.nn.init.orthogonal_(self.conv0.weight, 0.5)
        torch.nn.init.zeros_(self.conv0.bias)  

        torch.nn.init.orthogonal_(self.conv1.weight, 0.5)
        torch.nn.init.zeros_(self.conv1.bias)

        torch.nn.init.orthogonal_(self.conv2.weight, 0.5)
        torch.nn.init.zeros_(self.conv2.bias)

    
    #za.shape     = (batch_size, seq_length_a, features)
    #zb.shape     = (batch_size, seq_length_b, features)
    #return.shape = (batch_size, seq_length_a, seq_length_b)
    def forward(self, za, zb):  

        za_tile = za.unsqueeze(2)
        zb_tile = zb.unsqueeze(1)

        za_tile = za_tile.moveaxis(3, 1) 
        zb_tile = zb_tile.moveaxis(3, 1) 

        z_dif = za_tile - zb_tile

        y = self.conv0(z_dif)
        y = self.act0(y)
        
        y = self.conv1(y)
        y = self.act1(y)
        
        y = self.conv2(y)
        y = self.act2(y)

        y = y.squeeze(1)

        return y

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

        self.model_causality = ModelCausality(512, 128)

        for i in range(len(self.model)):
            if hasattr(self.model[i], "weight"):
                torch.nn.init.orthogonal_(self.model[i].weight, 2**0.5)
                torch.nn.init.zeros_(self.model[i].bias)

        

        print("model_target")
        print(self.model)
        print(self.model_causality)
        print("\n\n")

    def forward(self, x):
        return self.model(x)
    
    #z.shape = (batch, seq, features_count)
    #returns : (batch, seq, seq)
    def forward_causality(self, za, zb):
        return self.model_causality(za, zb)
    

if __name__ == "__main__":

    model = ModelCausality(512, 128)

    print(model)

    za = torch.randn((7, 3, 512))
    zb = torch.randn((7, 5, 512))

    y = model(za, zb)

    print(y.shape)