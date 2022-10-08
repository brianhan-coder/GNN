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

parser = argparse.ArgumentParser(description="Simulate a GNN with the appropriate hyperparameters.")
parser.add_argument('-d','--dataset', required=True, help='the protein dataset')
parser.add_argument('--graph_path', required=True, help='path to the graph files')
parser.add_argument('-r','--training_ratio', required=False, help='path to the pdb files',default=0.7)
parser.add_argument('--partition_ratio', required=False, type=str, help="governs the ration of partition sizes in the training, validation, and test sets. a list of the form [train, val, test]", default="0.4:0.3:0.3")
parser.add_argument('--partition_size', required=False, help='sets partition size for the total size of dataset', default='max')
parser.add_argument('-e','--epochs', required=False, help='number of training epochs', default='1001')
parser.add_argument('-n','--num_layers', required=False, help='number of additional layers, basic architecture has three', default='0')
parser.add_argument('-p','--patience', required=False, type=int, help='upper limit for the patience counter used in validation', default=20)
parser.add_argument('-b','--batch_size', required=False, type=int, help='batch size for training, testing and validation', default=40)
parser.add_argument('-l','--learning_rate', required=False, type=float, help='initial learning rate', default=0.01)
parser.add_argument('-m','--model_type', required=False, type=str, help='the underlying model of the neural network', default='GCN')
args = parser.parse_args()
protein_dataset=args.dataset
pdb_path=args.graph_path
partition_ratio=args.training_ratio
partition_size=args.partition_size
lr=args.learning_rate
n_epochs=args.epochs
arch=args.model_type
ratio = args.partition_ratio.split(":")
ratio = [float(entry) for entry in ratio]
batch_size=args.batch_size
num_layers=args.num_layers
if partition_size != 'max':
    parition_size = int(partition_size)

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
    graph_dataset=[]
    for protein_index,my_protein in enumerate(proteins):
        if os.path.exists(str(pdb_path)+'/'+str(my_protein)+".nx"):
            G = nx.read_gpickle(str(pdb_path)+'/'+str(my_protein)+".nx")
            graph_dataset.append(G)

    #print(graph_dataset[0].edge_attr)

    ### train test partition
    graph_dataset=GNN_core.balance_dataset(graph_dataset)
    GNN_core.get_info_dataset(graph_dataset,verbose=True)

    #train_test_partition=int(partition_ratio*len(graph_dataset))
    assert(ratio[0]+ratio[1]+ratio[2]==1)
    part1 = int(len(graph_dataset)*ratio[0])
    part2 = part1 + int(len(graph_dataset)*ratio[1])
    part3 = part2 + int(len(graph_dataset)*ratio[2])

    train_dataset = graph_dataset[:part1]
    test_dataset = graph_dataset[part1:part2]
    val_dataset = graph_dataset[part2:]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    print(f'Number of val graphs: {len(val_dataset)}')

    ### mini-batching of graphs, adjacency matrices are stacked in a diagonal fashion. Batching multiple graphs into a single giant graph

    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    ### core GNN 
    num_node_features=len(graph_dataset[0].x[0])
    num_classes=2
    hidden_channels=12
    if arch == 'GCN':
        model = GNN_core.GCN(hidden_channels,num_node_features=num_node_features,num_classes=num_classes,num_layers=num_layers)
    if arch == 'GNN':
        model = GNN_core.GNN(hidden_channels,num_node_features=num_node_features,num_classes=num_classes,num_layers=num_layers)
    if arch == 'GTN':
        model = GNN_core.GTN(hidden_channels,num_node_features=num_node_features,num_classes=num_classes,num_layers=num_layers)
    ### randomly initialize GCNConv model parameters
    for layer in model.children():
        if isinstance(layer, GCNConv):
            dic = layer.state_dict()
            for k in dic:
                dic[k] = torch.randn(dic[k].size())
            layer.load_state_dict(dic)
            del(dic)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0
    best_val_epoch = 0
    best_model=None

    ### training
    for epoch in range(1, int(n_epochs)):
  
        GNN_core.train(model=model,train_loader=train_loader,optimizer=optimizer,criterion=criterion)
        train_acc = GNN_core.test(model=model,loader=train_loader)
        test_acc = GNN_core.test(model=model,loader=test_loader)
        
        test_loss=GNN_core.loss(model=model,loader=test_loader,criterion=criterion).item()
        train_loss=GNN_core.loss(model=model,loader=train_loader,criterion=criterion).item()

        this_val_acc = GNN_core.test(model=model,loader=val_loader)
        if epoch %20==0:
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f},Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}')

        if this_val_acc > best_val_acc: #validation wrapper
            best_val_epoch = epoch
            best_val_acc=this_val_acc
            best_model= copy.deepcopy(model)
            patience_counter = 0
            print(f"new best validation score {best_val_acc}")
        else:
            patience_counter+=1
        if patience_counter == args.patience:
            print("ran out of patience")
            break
        
    trainscore = GNN_core.test(model=best_model,loader=train_loader)
    testscore = GNN_core.test(model=best_model,loader=test_loader)
    print(f'score on train set: {trainscore}')
    print(f'score on test set: {testscore}')
    predict_test = GNN_core.predict(model=best_model,loader=test_loader)
    label_test=[]
    for data in test_loader:
        label_test.append(data.y.tolist())

    label_test=[item for sublist in label_test for item in sublist]
    predict_test=[item for sublist in predict_test for item in sublist]
    #predict_test=np.array(predict_test).ravel()

    fpr1, tpr1, thresholds = roc_curve(label_test, predict_test)
    tn, fp, fn, tp = confusion_matrix(label_test, predict_test).ravel()
    AUROC = auc(fpr1, tpr1)
    print(f'  AUC: {AUROC}')
    print(f"  confusion matrix: [tn {tn}, fp {fp}, fn {fn}, tp {tp}]")
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    print(f'  precision = {precision}')
    print(f'  recall = {recall}')
    print(args)
    print(round(AUROC,3),round(trainscore,3),round(testscore,3),round(precision,3),round(recall,3),tn, fp, fn, tp)