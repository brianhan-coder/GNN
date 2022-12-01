from logging import exception
import os
from platform import architecture
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import networkx as nx
import GNN_core
import argparse
import random
from os.path import exists
from multiprocessing import Pool
import torch.optim as optim
import copy
from sklearn.metrics import confusion_matrix, roc_curve, auc

parser = argparse.ArgumentParser(description="Simulate a GNN with the appropriate hyperparameters.")
parser.add_argument('-d','--dataset', required=True, help='the protein dataset')
parser.add_argument('--graph_path', required=True, help='path to the graph files')
parser.add_argument('-r','--partition_ratio', required=False, type=str, help="governs the ration of partition sizes in the training, validation, and test sets. a list of the form [train, val, test]", default="0.4:0.3:0.3")
parser.add_argument('--partition_size', required=False, help='sets partition size for the total size of dataset', default='max')
parser.add_argument('-e','--epochs', required=False, help='number of training epochs', default='101')
parser.add_argument('-n','--num_layers', required=False, help='number of additional layers, basic architecture has three', default='3')
parser.add_argument('-nc','--num_c_layers', required=False, help='number of clustered layers', default='3')
parser.add_argument('-p','--patience', required=False, type=int, help='upper limit for the patience counter used in validation', default=20)
parser.add_argument('-b','--batch_size', required=False, type=int, help='batch size for training, testing and validation', default=30)
parser.add_argument('-l','--learning_rate', required=False, type=float, help='initial learning rate', default=0.008)
parser.add_argument('-m','--model_type', required=False, type=str, help='the underlying model of the neural network', default='GTN')
parser.add_argument('-c','--hidden_channel', required=False, type=int, help='width of hidden layers', default=25)
parser.add_argument('--cluster',action=argparse.BooleanOptionalAction, help='flag to cluster the amino acids')
args = parser.parse_args()
protein_dataset=args.dataset
pdb_path=args.graph_path
lr=args.learning_rate
n_epochs=args.epochs
arch=args.model_type
ratio = args.partition_ratio.split(":")
ratio = [float(entry) for entry in ratio]
batch_size=args.batch_size
num_layers=args.num_layers
num_c_layers=args.num_c_layers
hidden_channels=args.hidden_channel
cluster_flag=args.cluster
if args.cluster==None:
    cluster_flag=False

### load proteins
proteins_PDB=[]
proteins_Uniprot=[]
graph_labels=[]
with open(protein_dataset, "r") as file:
    content = file.read()
for line in content.splitlines():
    line=np.array(list(line.split(" ")))
    proteins_Uniprot.append(line[0])
    proteins_PDB.append(line[1])
    graph_labels.append(int(line[2]))

tmp = list(zip(proteins_Uniprot,proteins_PDB, graph_labels))   
random.shuffle(tmp) #shuffle the protein list


proteins_Uniprot,proteins_PDB, graph_labels = zip(*tmp)
proteins_Uniprot,proteins_PDB, graph_labels = list(proteins_Uniprot),list(proteins_PDB), list(graph_labels)
if args.partition_size != 'max':
    proteins_Uniprot=proteins_Uniprot[:int(args.partition_size)]
    proteins_PDB=proteins_PDB[:int(args.partition_size)]
    graph_labels=graph_labels[:int(args.partition_size)]

if __name__ == '__main__':
    ### loading graphs 
    graph_dataset=[]
    for protein_index,my_protein in enumerate(proteins_PDB):
        if os.path.exists(str(pdb_path)+'/'+str(my_protein)+".nx"):
            G = nx.read_gpickle(str(pdb_path)+'/'+str(my_protein)+".nx")
            G.prot_idx = torch.tensor(protein_index,dtype=torch.long)
            graph_dataset.append(G)

    graph_dataset=GNN_core.balance_dataset(graph_dataset)
    GNN_core.get_info_dataset(graph_dataset,verbose=True)

    assert(ratio[0]+ratio[1]+ratio[2]==1)
    part1 = int(len(graph_dataset)*ratio[0])
    part2 = part1 + int(len(graph_dataset)*ratio[1])
    part3 = part2 + int(len(graph_dataset)*ratio[2])
    ### core GNN 
    num_node_features=len(graph_dataset[0].x[0])
    num_classes=2

    if arch == 'GCN':
        model = GNN_core.GCN(hidden_channels,num_node_features=num_node_features,num_classes=num_classes,num_layers=num_layers)
    if arch == 'GNN':
        model = GNN_core.GNN(hidden_channels,num_node_features=num_node_features,num_classes=num_classes,num_layers=num_layers)
    if arch == 'GTN' and cluster_flag==False:
        model = GNN_core.GTN(hidden_channels,num_node_features=num_node_features,num_classes=num_classes,num_layers=num_layers)
    if arch == 'GTN' and cluster_flag==True:
        model = GNN_core.GTN_hybrid(hidden_channels,num_node_features=num_node_features,num_classes=num_classes,num_layers=num_layers,num_c_layers=num_c_layers)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    ### mini-batching of graphs, adjacency matrices are stacked in a diagonal fashion. Batching multiple graphs into a single giant graph
    train_dataset = graph_dataset[:part1]
    test_dataset = graph_dataset[part1:part2]
    val_dataset = graph_dataset[part2:]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    print(f'Number of val graphs: {len(val_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    ### clustering
    if cluster_flag==True:
        clustered_train_dataset=GNN_core.clustering_graph(train_dataset,proteins_Uniprot)
        clustered_test_dataset=GNN_core.clustering_graph(test_dataset,proteins_Uniprot)
        clustered_val_dataset=GNN_core.clustering_graph(val_dataset,proteins_Uniprot)
        clustered_train_loader = [DataLoader(train_dataset, batch_size=batch_size, shuffle=False),DataLoader(clustered_train_dataset, batch_size=batch_size, shuffle=False)]
        clustered_test_loader = [DataLoader(test_dataset, batch_size=batch_size, shuffle=False),DataLoader(clustered_test_dataset, batch_size=batch_size, shuffle=False)]
        clustered_val_loader = [DataLoader(val_dataset, batch_size=batch_size, shuffle=False),DataLoader(clustered_val_dataset, batch_size=batch_size, shuffle=False)]

    best_val_acc = 0
    best_val_epoch = 0
    best_model=None

    ### training
    for epoch in range(1, int(n_epochs)):

        if cluster_flag==True:
            GNN_core.train_hybrid(model=model,train_loader=clustered_train_loader,optimizer=optimizer,criterion=criterion)
            train_acc = GNN_core.test_hybrid(model=model,loader=clustered_train_loader)
            test_acc = GNN_core.test_hybrid(model=model,loader=clustered_test_loader)
            this_val_acc = GNN_core.test_hybrid(model=model,loader=clustered_val_loader)
            test_loss=GNN_core.loss_hybrid(model=model,loader=clustered_test_loader,criterion=criterion).item()
            train_loss=GNN_core.loss_hybrid(model=model,loader=clustered_train_loader,criterion=criterion).item()
        else:
            GNN_core.train(model=model,train_loader=train_loader,optimizer=optimizer,criterion=criterion)
            train_acc = GNN_core.test(model=model,loader=train_loader)
            test_acc = GNN_core.test(model=model,loader=test_loader)
            this_val_acc = GNN_core.test(model=model,loader=val_loader)
            test_loss=GNN_core.loss(model=model,loader=test_loader,criterion=criterion).item()
            train_loss=GNN_core.loss(model=model,loader=train_loader,criterion=criterion).item()

        if epoch %20==0:
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f},Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}')

        if this_val_acc > best_val_acc: #validation wrapper
            best_val_epoch = epoch
            best_val_acc=this_val_acc
            best_model= copy.deepcopy(model)
            patience_counter = 0
            #print(f"new best validation score {best_val_acc}")
        else:
            patience_counter+=1
        if patience_counter == args.patience:
            print("ran out of patience")
            break
    if cluster_flag==True: 
        trainscore = GNN_core.test_hybrid(model=best_model,loader=clustered_train_loader)
        testscore = GNN_core.test_hybrid(model=best_model,loader=clustered_test_loader)
        predict_test = GNN_core.predict_hybrid(model=best_model,loader=clustered_test_loader)
    else:
        trainscore = GNN_core.test(model=best_model,loader=train_loader)
        testscore = GNN_core.test(model=best_model,loader=test_loader)
        predict_test = GNN_core.predict(model=best_model,loader=test_loader)

    print(f'score on train set: {trainscore}')
    print(f'score on test set: {testscore}')


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