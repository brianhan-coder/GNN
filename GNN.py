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
from multiprocessing import Pool
import multiprocessing


#创建输入命令
parser = argparse.ArgumentParser(description="Simulate a GNN with the appropriate hyperparameters.")
parser.add_argument('-d','--dataset', required=True, help='the protein dataset')
parser.add_argument('--pdb_path', required=True, help='path to the pdb files')#PDB？
parser.add_argument('-r','--training_ratio', required=False, help='path to the pdb files',default=0.7)#parameter
parser.add_argument('--partition_size', required=False, help='sets partition size for the total size of dataset', default='max')
parser.add_argument('-e','--epochs', required=False, help='number of training epochs', default='10')
args = parser.parse_args()
protein_dataset=args.dataset
pdb_path=args.pdb_path
partition_ratio=args.training_ratio
partition_size=args.partition_size
n_epochs=args.epochs
if partition_size != 'max':
    parition_size = int(partition_size)

gnn_layer_by_name = {"GCN": geom_nn.GCNConv, "GAT": geom_nn.GATConv, "GraphConv": geom_nn.GraphConv}#函数索引

### loading features

featureData = feature_embedding.FeatureData()
#featureData.readFeatureFile("AA_features/AA_vanderWaalsVolume.dat","vdWVolume")
#featureData.readFeatureFile("AA_features/AA_GraphShapeIndex.dat","shape")
#featureData.readFeatureFile("AA_features/AA_NHydrogenBondDonors.dat","N_HBond")
#featureData.readFeatureFile("AA_features/AA_MolecularMass.dat","mass")
featureData.readFeatureFile("AA_features/AA_polarity.dat","polarity")
featureData.readFeatureFile("AA_features/AA_hydrophobicity.dat","hydrophobicity")
featureData.readFeatureFile("AA_features/AA_flexibility.dat","flexibility")
featureData.readFeatureFile("AA_features/AA_IDP_Scale.dat","IDP_scale")
featureData.readFeatureFile("AA_features/AA_charge.dat","charge")
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


if __name__ == '__main__':
    ### parallel converting PDB to graphs 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #Device Info
    
    input_list=[]
    for protein_index,my_protein in enumerate(proteins):
        input_list.append([pdb_path,my_protein,featureData,graph_labels,protein_index])

    pool=multiprocessing.Pool()
    print('number of workers: ',pool._processes)
    graph_dataset=pool.map(GNN_core.convert_pdb2graph,input_list)
    pool.close()
    pool.join()
    graph_dataset = list(filter(lambda item: item is not None, graph_dataset))
    #print(graph_dataset[0].edge_attr)

    ### train test partition
    graph_dataset=GNN_core.balance_dataset(graph_dataset)
    GNN_core.get_info_dataset(graph_dataset,verbose=True)
    train_test_partition=int(partition_ratio*len(graph_dataset))
    train_dataset = graph_dataset[:train_test_partition]
    test_dataset = graph_dataset[train_test_partition:]
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    ### mini-batching of graphs, adjacency matrices are stacked in a diagonal fashion. Batching multiple graphs into a single giant graph

    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    ### core GNN 
    num_node_features=len(graph_dataset[0].x[0])
    num_classes=2
    model = GNN_core.GCN(hidden_channels=64,num_node_features=num_node_features,num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    ### training
    for epoch in range(1, int(n_epochs)):
        GNN_core.train(model=model,train_loader=train_loader,optimizer=optimizer,criterion=criterion)  
        train_acc = GNN_core.test(model=model,loader=train_loader)
        test_acc = GNN_core.test(model=model,loader=test_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

