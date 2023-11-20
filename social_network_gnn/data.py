import json
import collections
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
# from torch_geometric.transforms import AddTrainValTestMask as masking
from torch_geometric.utils.convert import to_networkx
from torch_geometric.nn import GCNConv

from social_network_gnn import CONFIG
from pathlib import Path

import networkx as nx

class Graph:
    def __init__(self, dataset_path: Path):
        breakpoint()
        edge_name = list(dataset_path.glob("*_edges.csv"))[0]
        feature_name = list(dataset_path.glob("*.json"))[0]
        target_name = list(dataset_path.glob("*.json"))[0]
        
        with open(feature_name) as f:
            self.features = json.load(f)
            self.features = {int(k): int(v) for k, v in self.features.items()}
        self.edges = pd.read_csv(edge_name)
        self.targets = pd.read_csv(target_name)

    def get_edges(self):
        pass

    def numpy(self):
        pass

    def torch(self):
        node_features=list(self.features.values())
        node_features=torch.tensor(node_features)
        node_labels=torch.tensor(self.targets['ml_target'].values)
        edges_list=self.edges.values.tolist()

        # Add double edges since our graph contains undirected edges
        edge_index01=torch.tensor(edges_list, dtype = torch.long).T
        edge_index02=torch.zeros(edge_index01.shape, dtype = torch.long)
        edge_index02[0,:]=edge_index01[1,:]
        edge_index02[1,:]=edge_index01[0,:]
        edge_index0=torch.cat((edge_index01,edge_index02),axis=1)
        return Data(x=node_features, y=node_labels, edge_index=edge_index0)