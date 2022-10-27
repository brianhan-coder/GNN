from logging import exception
import os
from platform import architecture
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_geometric.nn as geom_nn
from torch_geometric.nn import GCNConv
import networkx as nx
import feature_embedding
import PDB2Graph
import GNN_core
import argparse
import random
from os.path import exists
from multiprocessing import Pool
import multiprocessing
import torch.optim as optim
import copy
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import GNN_clustering as cluster
import igraph as ig

def contact_map_plot(first,second,num_nodes):
    contact=np.zeros((num_nodes,num_nodes))
    for i in range(len(first)):
        if first[i]>second[i]:
            a=first[i]
            b=second[i]
        else:
            b=first[i]
            a=second[i]
        contact[b][a]=1
    fig, ax = plt.subplots()
    im = ax.imshow(contact)
    plt.show()

def contact_map(first,second,num_nodes):
    contact=np.zeros((num_nodes,num_nodes))
    for i in range(len(first)):
        a=first[i]
        b=second[i]
        contact[b][a]=1
        contact[a][b]=1
    return contact

def spectral_cluster(first,second,num_nodes):
    A=contact_map(first,second,num_nodes)
    # diagonal matrix
    D = np.diag(A.sum(axis=1))
    # graph laplacian
    L = D-A
    # eigenvalues and eigenvectors
    vals, vecs = np.linalg.eig(L)

     # sort these based on the eigenvalues
    vecs = vecs[:,np.argsort(vals)]
    vals = vals[np.argsort(vals)]   

    clusters=[20]
    best_asy=1.
    for n_cluster in clusters:
        kmeans = KMeans(n_clusters=n_cluster)
        kmeans.fit(vecs[:,1:n_cluster])
        node_label=kmeans.labels_
        node_label=((node_label-min(node_label))/(max(node_label)-min(node_label))-0.5)*2
        contact_cluster=np.zeros((num_nodes,num_nodes))
        assymetry=0.
        total_count=0
        for i in range(num_nodes):
            for j in range(i,num_nodes):
                if A[i][j]!=0:
                    contact_cluster[i][j]=node_label[i]
                    contact_cluster[j][i]=node_label[j]
                    if node_label[i]!=node_label[j]:
                        assymetry+=1.
                    total_count+=1.
        asy_degree=assymetry/total_count/n_cluster
        #print('current number of clusters:', n_cluster, 'current asy degree: ',asy_degree)
        if asy_degree<best_asy:
            best_cluster=contact_cluster
            best_asy=asy_degree
            best_n_cluster=n_cluster
            best_node_label=node_label
    #print('best number of clusters: ',best_n_cluster, 'best degree of asymmetry: ',best_asy)


    #sequence_region=np.zeros((10,num_nodes))
    #for i in range(num_nodes):
    #    for j in range(10):
    #        sequence_region[j][i]=best_node_label[i]
    #fig, ax = plt.subplots()
    #im = ax.imshow(sequence_region,cmap='RdBu')
    #plt.show()
    fig, ax = plt.subplots()
    im = ax.imshow(best_cluster,cmap='RdBu')
    plt.show()


def numberEdge_perNode(first,second,num_nodes):
    node_by_res=[0]*num_nodes
    for i in range(0,len(first)):
        node_by_res[first[i]]+=1
        node_by_res[second[i]]+=1
    plt.plot(range(0,num_nodes),node_by_res, marker='o', color='red')
    plt.show()

parser = argparse.ArgumentParser(description="Simulate a GNN with the appropriate hyperparameters.")
parser.add_argument('-d','--dataset', required=True, help='the protein dataset')
parser.add_argument('-g','--graph_path', required=True, help='path to the graph files')
parser.add_argument('--partition_size', required=False, help='sets partition size for the total size of dataset', default='max')

args = parser.parse_args()
protein_dataset=args.dataset
pdb_path=args.graph_path
partition_size=args.partition_size

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
index=0
proteins[0]='3W06'
for protein_index,my_protein in enumerate(proteins):
    if os.path.exists(str(pdb_path)+'/'+str(my_protein)+".nx") and index<1:
        G = nx.Graph(nx.read_gpickle(str(pdb_path)+'/'+str(my_protein)+".nx"))
        G_new=nx.Graph()
        num_nodes=len(list(G['x'])[0].numpy())
        first=list(G['edge_index'])[0].numpy()[0]
        second=list(G['edge_index'])[0].numpy()[1]
        print('protein: ',my_protein,'edge/res: ',2*len(first)/num_nodes,num_nodes)
        #contact_map_plot(first,second,num_nodes)
        #numberEdge_perNode(first,second,num_nodes)
        spec_c=cluster.spectral_cluster(G)
        modul_c=cluster.modularity_clustering(G)

        index+=1
        '''
        node_feature=list(G['x'])[0].numpy()
        node_name_dic={}
        node_type=[-1]*num_nodes
        j=0
        for i in range(0,num_nodes):
            if str(node_feature[i]) not in node_name_dic.keys():
                node_name_dic[str(node_feature[i])]=j
                node_type[i]=j
                j+=1
            
            else:
                node_type[i]=node_name_dic[str(node_feature[i])]

        nodes=range(0,num_nodes)
        node_color=[]
        for i in range(0,len(first)):
            if first[i] in nodes or second[i] in nodes:
                G_new.add_edge(first[i],second[i])
        for no in G_new.nodes():
            node_color.append(node_type[no])

        print('protein: ',my_protein,'edge/res: ',2*len(first)/num_nodes,num_nodes)
        options = {"font_size": 0,"node_size": 60,"node_color": node_color,"edgecolors": "black","linewidths": 1,"width": 1,}
        #nx.draw_networkx(G_new, **options,cmap=plt.cm.RdYlGn)
        #ax = plt.gca()
        #ax.margins(0.20)
        #plt.axis("off")
        #plt.show()
        
        index+=1
        '''

