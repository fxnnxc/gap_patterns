
class BaseModel():
    def __init__(self, flags):
        self.flags = flags 
        

import numpy as np 
import torch 
import torch.nn as nn
        
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def make_fc(in_features, hidden_dim, num_layers, activation, out_features):
    if activation == "relu":
        ACT = nn.ReLU
    elif activation == "tanh":
        ACT = nn.Tanh
    else:
        raise NotImplementedError()
    net = []
    if num_layers == 1:
        net.append(layer_init(nn.Linear(in_features, out_features)))
    else:
        net.append(layer_init(nn.Linear(in_features, hidden_dim)))
        net.append(ACT()) 
    
        for idx in range(num_layers-2):
            net.append(layer_init(nn.Linear(hidden_dim, hidden_dim)))
            net.append(ACT())
        net.append(layer_init(nn.Linear(hidden_dim, out_features)))
    net.append(ACT())
    return nn.Sequential(*net)
