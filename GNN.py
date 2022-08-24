from logging import exception
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
import argparse
import random
from os.path import exists

parser = argparse.ArgumentParser(description="Simulate a GNN with the appropriate hyperparameters.")
parser.add_argument('-d','--dataset', required=True, help='the protein dataset')
parser.add_argument('--pdb_path', required=True, help='path to the pdb files')
parser.add_argument('-r','--training_ratio', required=False, help='path to the pdb files',default=0.7)
parser.add_argument('--partition_size', required=False, help='sets partition size for the total size of dataset', default='max')

args = parser.parse_args()
protein_dataset=args.dataset
pdb_path=args.pdb_path
partition_ratio=args.training_ratio
partition_size=args.partition_size
if partition_size != 'max':
    parition_size = int(partition_size)

gnn_layer_by_name = {"GCN": geom_nn.GCNConv, "GAT": geom_nn.GATConv, "GraphConv": geom_nn.GraphConv}

### loading features
featureData = feature_embedding.FeatureData()
featureData.readFeatureFile("AA_features/AA_vanderWaalsVolume.dat","vdWVolume")
featureData.readFeatureFile("AA_features/AA_GraphShapeIndex.dat","shape")
featureData.readFeatureFile("AA_features/AA_polarity.dat","polarity")
featureData.readFeatureFile("AA_features/AA_hydrophobicity.dat","hydrophobicity")
featureData.readFeatureFile("AA_features/AA_NHydrogenBondDonors.dat","N_HBond")
featureData.readFeatureFile("AA_features/AA_MolecularMass.dat","mass")

### load proteins
proteins=[]
graph_labels=[]
with open(protein_dataset, "r") as file:
    content = file.read()
for line in content.splitlines():
    line=np.array(list(line.split("\t")))
    proteins.append(line[0])
    graph_labels.append(int(line[1]))

tmp = list(zip(proteins, graph_labels))
random.shuffle(tmp)
proteins, graph_labels = zip(*tmp)
proteins, graph_labels = list(proteins), list(graph_labels)

if partition_size != 'max':
    proteins=proteins[:int(partition_size)]
    graph_labels=graph_labels[:int(partition_size)]

graph_dataset=[]
for protein_index,my_protein in enumerate(proteins):
    ### loading PDB for the protein structure if it exists
    if exists(str(pdb_path)+'/'+str(my_protein)+'.pdb'):
        try:
            nodes=PDB2Graph.nodes(pdb_path,my_protein)
            edges=PDB2Graph.edges(pdb_path,my_protein)
            print("Loaded ",str(my_protein))
        except (IndexError, ValueError):
            print("Can't load ",str(my_protein))
            continue
        
    ### readin feature list of all amino acids
        try:
            complete_list_feature=[]
            for aminoAcid in featureData.buildProtein(nodes):
                my_features=[float(tmp) for tmp in list(aminoAcid)]
                complete_list_feature.append(my_features)

            nodes_features=torch.tensor(complete_list_feature,dtype=torch.float) # feature vector of all nodes
            edge_index = torch.tensor(edges, dtype=torch.long) # edges, 1st list: index of the source nodes, 2nd list: index of target nodes.
            my_label=torch.tensor(graph_labels[protein_index], dtype=torch.long)
            g = Data(x=nodes_features, edge_index=edge_index,y=my_label)
            graph_dataset.append(g)
        except KeyError:
            continue
### train test partition

train_test_partition=int(partition_ratio*len(graph_dataset))
train_dataset = graph_dataset[:train_test_partition]
test_dataset = graph_dataset[train_test_partition:]
print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

### mini-batching of graphs, adjacency matrices are stacked in a diagonal fashion. Batching multiple graphs into a single giant graph

from torch_geometric.loader import DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

### core GNN 
num_node_features=len(graph_dataset[0].x[0])
num_classes=2
model = GNN_core.GCN(hidden_channels=64,num_node_features=num_node_features,num_classes=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

### training
for epoch in range(1, 5):
    GNN_core.train(model=model,train_loader=train_loader,optimizer=optimizer,criterion=criterion)
    train_acc = GNN_core.test(model=model,loader=train_loader)
    test_acc = GNN_core.test(model=model,loader=test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')