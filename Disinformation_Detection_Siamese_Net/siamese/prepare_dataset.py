import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np


def match_triplets(anchor_i, intersect_on, dataset, n_triplets=10):
    # resample the dataset to avoid getting the same samples
    anchor = intersect_on[anchor_i]
    rnd_indices = torch.randperm(intersect_on.size(0))
    intersect_on = intersect_on[rnd_indices]
    anchor_intersections = torch.sum(intersect_on * anchor, dim=1)/anchor.size(0)
    #anchor = anchor[None,:]
    
    positives = torch.nonzero(anchor_intersections)[:n_triplets, 0]
    negatives = torch.nonzero(anchor_intersections == 0)[:n_triplets, 0]
    n_triplets = min(positives.size(0), negatives.size(0))
    positives, negatives = positives[:n_triplets], negatives[:n_triplets]
    anchors = dataset[anchor_i].repeat(n_triplets, 1)
    #print(anchors, negatives.size())
    return torch.stack((anchors, dataset[positives], dataset[negatives]), dim=-1)


def triplet_distance(anchors, poss, negs):
    pos_distances = F.pairwise_distance(anchors, poss)
    neg_distances = F.pairwise_distance(anchors, negs)
    return pos_distances - neg_distances


class MonthDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, n_anchors, entity_embs, batch_size=64, n_worst=14, n_best=50):
        anchor_indexes = torch.randint(0, len(dataset), size=(n_anchors,))
        
        anchor_triplets = [match_triplets(i, entity_embs, dataset) for i in anchor_indexes] 
        self.data = torch.cat(anchor_triplets, dim=0)
        print(self.data.size())
      

    def __getitem__(self, index):
        return self.data[index]
        
    def __len__(self):
        return len(self.data)


class TripletBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data, batch_size=64, n_worst=14, n_rand=50):
        self.batch_size = batch_size
        self.n_worst = n_worst
        self.n_rand = n_rand
        self.data = data
        self.triplet_distances = triplet_distance(*(data.T.mT))
        
    def update_distances(self, new_distances):
        if len(new_distances) == len(self.data):
            self.triplet_distances = new_distances

    def __iter__(self):
        _, sorted_distances = torch.sort(self.triplet_distances)
        n_batches = int(np.ceil(len(self.data)/self.batch_size))
        
        for i in range(n_batches):
            rand_indices = torch.randint(n_batches*self.n_worst, len(self.data), size=(self.n_rand, )) 
            worst_indices = torch.randint(i*self.n_worst, (i+1)*self.n_worst, size=(self.n_worst, ))
            sample_indices = torch.cat((rand_indices, worst_indices), dim=0)
           # print(sample_indices)
            yield sample_indices

    def __len__(self):
        return len(self.data)