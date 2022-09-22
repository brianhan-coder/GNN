from torch.nn import Linear
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch
from torch_geometric.nn import GraphConv
import random
import numpy as np
from os.path import exists
import PDB2Graph
from torch_geometric.data import Data
from proteingraph.pin import pdb2df
from proteingraph import read_pdb
import GNN_core

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
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch,hidden_channels,num_layers):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.bn1(x)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.bn2(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)

        # 1.1 Additional deep layers
        for l in range(int(num_layers)):
            x=x.GCNConv(hidden_channels, hidden_channels)
            x=x.torch.nn.BatchNorm1d(hidden_channels)
            x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

### helper function to output the internal activations
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def train(model,train_loader,optimizer,criterion,hidden_channels,num_layers):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        #model.conv1.register_forward_hook(get_activation('conv3'))
        out = model(data.x, data.edge_index, data.batch,hidden_channels,num_layers)  # Perform a single forward pass.
        #print(np.mean(activation['conv3'].numpy(),axis=0))
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(model,loader,hidden_channels,num_layers):
     model.eval()
     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch,hidden_channels,num_layers)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.

def predict(model,loader):
    output=[]
    model.eval()
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.numpy()
        output.append(pred)
    return output


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


def convert_pdb2graph(input):
 
    pdb_path=input[0]
    my_protein=input[1]
    featureData=input[2]
    graph_labels=input[3]
    protein_index=input[4]
    path=str(pdb_path)+'/'+str(my_protein)+'.pdb'
    if exists(path):
        try:
            pdb_df=pdb2df(path)
            res2node=PDB2Graph.residueID2nodeID(pdb_df)
            G=read_pdb(str(pdb_path)+'/'+str(my_protein)+'.pdb')
            PDB2Graph.add_pi_pi_interactions(G,pdb_df)

            node=PDB2Graph.node(G,res2node)
            edge=PDB2Graph.edge(G,res2node)
            print("Loaded ",str(my_protein))
        except (IndexError, ValueError):
            print("Can't load ",str(my_protein))
            return

        
    ### readin feature list of all amino acids
        try:
            complete_list_feature=[]
            for aminoAcid in featureData.buildProtein(node):
                my_features=[float(tmp) for tmp in list(aminoAcid)]
                complete_list_feature.append(my_features)

            nodes_features=torch.tensor(complete_list_feature,dtype=torch.float) # feature vector of all nodes
            edge_index = torch.tensor(edge, dtype=torch.long) # edges, 1st list: index of the source nodes, 2nd list: index of target nodes.
            my_label=torch.tensor(graph_labels[protein_index], dtype=torch.long)
            g = Data(x=nodes_features, edge_index=edge_index,y=my_label)

            return g
        except KeyError:
            print("Failed loading aminoacids info for ",str(my_protein))
            return

