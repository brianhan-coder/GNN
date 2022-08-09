import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_geometric.nn as geom_nn
import networkx as nx
import feature_embedding
import PDB2Graph
import GNN_core


gnn_layer_by_name = {"GCN": geom_nn.GCNConv, "GAT": geom_nn.GATConv, "GraphConv": geom_nn.GraphConv}

### loading features
featureData = feature_embedding.FeatureData()
featureData.readFeatureFile("AA_features/AA_charge.dat","charge")
featureData.readFeatureFile("AA_features/AA_hydrophobic.dat","hydrophobicity")


dataset=[]
proteins=['2VIU']
graph_labels=[[1]]
for protein_index,my_protein in enumerate(proteins):
### loading PDB for the protein structure
    nodes=PDB2Graph.nodes(my_protein)
    edges=PDB2Graph.edges(my_protein)

### readin feature list of all amino acids
    complete_list_feature=[]
    for aminoAcid in featureData.buildProtein(nodes):
        my_features=[float(tmp) for tmp in list(aminoAcid)]
        complete_list_feature.append(my_features)

    nodes_features=torch.tensor(complete_list_feature,dtype=torch.float) # feature vector of all nodes
    edge_index = torch.tensor(edges, dtype=torch.long) # edges, 1st list: index of the source nodes, 2nd list: index of target nodes.
    my_label=torch.tensor(graph_labels[protein_index], dtype=torch.long)
    g = Data(x=nodes_features, edge_index=edge_index,y=my_label)
    dataset.append(g)

### train test partition
#dataset = dataset.shuffle()
partition_ratio=1
train_test_partition=int(partition_ratio*len(dataset))
train_dataset = dataset[:train_test_partition]
test_dataset = dataset[train_test_partition:]
print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

### mini-batching of graphs, adjacency matrices are stacked in a diagonal fashion. Batching multiple graphs into a single giant graph

from torch_geometric.loader import DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

### core GNN 
num_node_features=len(dataset[0].x[0])
num_classes=2
model = GNN_core.GCN(hidden_channels=64,num_node_features=num_node_features,num_classes=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

### training
for epoch in range(1, 2):
    GNN_core.train(model=model,train_loader=train_loader,optimizer=optimizer,criterion=criterion)
    train_acc = GNN_core.test(model=model,loader=train_loader)
    #test_acc = GNN_core.test(model=model,loader=test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}')
    #print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')