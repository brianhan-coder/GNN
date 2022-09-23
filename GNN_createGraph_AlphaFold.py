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


parser = argparse.ArgumentParser(description="Simulate a GNN with the appropriat0;95;0ce hyperparameters.")
parser.add_argument('-d','--dataset', required=True, help='the protein dataset')
parser.add_argument('--pdb_path', required=True, help='path to the pdb files')
parser.add_argument('-r','--training_ratio', required=False, help='path to the pdb files',default=0.7)
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

gnn_layer_by_name = {"GCN": geom_nn.GCNConv, "GAT": geom_nn.GATConv, "GraphConv": geom_nn.GraphConv}

### loading features

featureData = feature_embedding.FeatureData()
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
    line=np.array(list(line.split(" ")))
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
    for protein_index,my_protein in enumerate(proteins):
        input_list=[pdb_path,my_protein,featureData,graph_labels,protein_index]

        G=GNN_core.convert_pdb2graph(input_list)
        if G!=None:
            nx.write_gpickle(G, "graph_base/graphAll_AlphaFold/"+str(my_protein)+".nx")
