import torch.nn.functional as F 
import torch 
import torch.nn as nn

from .base_model import make_fc


class SimpleGAP(nn.Module):
    def __init__(self, flags):
        super().__init__()
        self.one_hop_dims = flags.encoder.one_hop_dims 
        self.proj_dim = flags.cnn_dim
        self.proj_hop    = make_fc(in_features=sum(self.one_hop_dims), 
                              hidden_dim=self.proj_dim, 
                              num_layers=3,
                              activation='relu', 
                              out_features=self.proj_dim)
        
        self.indices = self.one_hop_dims[:]
        for i in range(len(self.one_hop_dims)-1):
            self.indices[i+1] += self.indices[i]
        self.indices = self.indices[:-1]
        
    def forward(self, x):
        temps = []
        for y in x:
            for s in y:
                temps.append(F.adaptive_avg_pool2d(s, output_size=1).view(s.size(0), -1))
        x = torch.cat(temps, dim=-1)
        x = self.proj_hop(x)
        return x


if __name__ == "__main__":
    encoder = None 