from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch
from torch_geometric.nn import GraphConv
import random
import numpy as np

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels,num_node_features,num_classes):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)  
        self.conv3 = GraphConv(hidden_channels, hidden_channels)  
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels,num_node_features,num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

def train(model,train_loader,optimizer,criterion):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(model,loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


def get_info_dataset(dataset, verbose=False):
    """Determines the number of inputs labeled one and zero in a dataset."""
    zeros = 0
    ones = 0
    for data in dataset:
        label = data.y.item()
        if label == 0:
            zeros+=1
        elif label ==1:
            ones+=1
    if verbose:
        print(f'In this dataset, there are {zeros} inputs labeled "0" and {ones} inputs labeled "1".')
    return (ones, zeros)

def balance_dataset(dataset):
    ones, zeros = get_info_dataset(dataset)
    if zeros==ones:
        return dataset
    if zeros>ones:
        major=zeros
        minor=ones
        the_major_one=0
    else:
        major=ones
        minor=zeros
        the_major_one=1

    ratio = float((major-minor)/(major))
    balanced = []
    for item in dataset:
        label = item.y.item()
        if label == the_major_one:
            if random.random()>ratio:
                balanced.append(item)
        else:
            balanced.append(item)
    return balanced