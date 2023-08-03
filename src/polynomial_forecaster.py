from base_forecaster import Forecaster
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolynomialForecaster(Forecaster):

    def __init__(self, num_degrees, bias=True):
        super().__init__()
        #transforms from (B,num_samples) to (B,degree,num_samples)
        self.power = lambda x: torch.cat([torch.pow(x,deg) for deg in range(0,num_degrees+1)] if bias else [torch.pow(x,deg+1) for deg in range(1, num_degrees+1)])
    
    def forward(self, x, coefs):
        x = self.power(x)
        #Coefs must either have shape (B, degrees, 1) or (1, degrees, 1)
        assert len(coefs.shape)==3 and coefs.shape[1]==x.shape[0], ('bad shape')
        out = torch.sum(coefs * x,dim=1)
        return out
    

# class Cubic

# if __name__ == '__main__':
#     coefs = nn.Parameter(torch.tensor([.5, 1.2, 2.5, 1.0], requires_grad=True).view(1,-1,1))
#     poly=PolynomialForecaster(3,True)

#     optim = torch.optim.AdamW([coefs], lr=.001)
    
#     t = torch.linspace(-10,10, 1000).unsqueeze(0)
#     y = torch.pow(t, 3) + torch.pow(t, 2) + t + 1
#     losses = torch.tensor([])
#     for i in range(50000):
#         y_pred = poly(t, coefs)
#         loss = nn.MSELoss()(y_pred, y)
    
#         optim.zero_grad()
#         loss.backward()
#         losses = torch.cat((losses, torch.tensor([loss])))
#         optim.step()
#         if i % 1000 == 0:
#             print(f'epoch {i}: loss {sum(losses[-500:]/500)} : coefs {coefs.float()}')
