import torch
import torch.nn as nn
import torch.nn.functional as F
from .prepare_dataset import triplet_distance


class TripletSiamese(nn.Module):
    def __init__(self, input_size=175):
        super(TripletSiamese, self).__init__()
        self.dnn1 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),    
            nn.Linear(128, 128),
            nn.ReLU(True),   
            nn.Linear(128, input_size)
        )

    def forward(self, x):
        return self.dnn(x)
        
    def forward(self, a, p, n):
        out_a = self.dnn1(a)
        out_p = self.dnn1(p)
        out_n = self.dnn1(n)
        return out_a, out_p, out_n

class TripletLoss(torch.nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()
        
    def forward(self, anchor, p, n):
        # clip(dist, min=0) is equal to max(dist, 0)
        return torch.clip(triplet_distance(anchor, p, n) + 1e-3, min=0)