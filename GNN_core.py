from torch.nn import Linear
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import global_mean_pool
import torch
from torch_geometric.nn import GraphConv,TransformerConv,GCNConv
import random
import numpy as np
import networkx as nx
from os.path import exists
import PDB2Graph
from torch_geometric.data import Data
from proteingraph.pin import pdb2df
from proteingraph import read_pdb
import GNN_clustering
import time
import copy

class GTN_hybrid(torch.nn.Module):
    def __init__(self, hidden_channels,num_node_features,num_classes,num_layers,num_c_layers):
        super(GTN_hybrid, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = TransformerConv(num_node_features, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv=torch.nn.ModuleList()
        self.bn=torch.nn.ModuleList()
        for l in range(int(num_layers)):
            self.conv.append(TransformerConv(hidden_channels, hidden_channels))
            self.bn.append(torch.nn.BatchNorm1d(hidden_channels))

        self.conv_c=torch.nn.ModuleList()
        self.bn_c=torch.nn.ModuleList()
        for l in range(int(num_c_layers)):
            self.conv_c.append(TransformerConv(hidden_channels, hidden_channels))
            self.bn_c.append(torch.nn.BatchNorm1d(hidden_channels))

        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index[0])
        x = self.bn1(x)
        x = x.relu()

        if len(self.conv_c) > 0:
            for index,conv_c_i in enumerate(self.conv_c):
                x = conv_c_i(x,edge_index[1])
                x = self.bn_c[index](x)
                x = x.relu()

        if len(self.conv) > 0:
            for index,conv_i in enumerate(self.conv):
                x = conv_i(x,edge_index[0])
                x = self.bn[index](x)
                x = x.relu()


        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        x = self.lin(x)
        return x

class GTN(torch.nn.Module):
    def __init__(self, hidden_channels,num_node_features,num_classes,num_layers):
        super(GTN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = TransformerConv(num_node_features, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = TransformerConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv=torch.nn.ModuleList()
        self.bn=torch.nn.ModuleList()
        for l in range(int(num_layers)):
            self.conv.append(TransformerConv(hidden_channels, hidden_channels))
            self.bn.append(torch.nn.BatchNorm1d(hidden_channels))

        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = x.relu()
        # 1.1 Additional deep layers
        if len(self.conv) > 0:
            for index,conv_i in enumerate(self.conv):
                x = conv_i(x,edge_index)
                x = self.bn[index](x)
                x = x.relu()
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        x = self.lin(x)
        return x

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels,num_node_features,num_classes,num_layers):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv=torch.nn.ModuleList()
        self.bn=torch.nn.ModuleList()
        for l in range(int(num_layers)):
            self.conv.append(GraphConv(hidden_channels, hidden_channels))
            self.bn.append(torch.nn.BatchNorm1d(hidden_channels))

        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = x.relu()
        # 1.1 Additional deep layers
        if len(self.conv) > 0:
            for index,conv_i in enumerate(self.conv):
                x = conv_i(x,edge_index)
                x = self.bn[index](x)
                x = x.relu()
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        x = self.lin(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels,num_node_features,num_classes,num_layers):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv=torch.nn.ModuleList()
        self.bn=torch.nn.ModuleList()
        for l in range(int(num_layers)):
            self.conv.append(GCNConv(hidden_channels, hidden_channels))
            self.bn.append(torch.nn.BatchNorm1d(hidden_channels))
       

        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = x.relu()
        # 1.1 Additional deep layers
        if len(self.conv) > 0:
            for index,conv_i in enumerate(self.conv):
                x = conv_i(x,edge_index)
                x = self.bn[index](x)
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

def train_hybrid(model,train_loader,optimizer,criterion):
    model.train()
    for data_old, data_new in zip(train_loader[0], train_loader[1]):
        #model.conv1.register_forward_hook(get_activation('conv3'))

        data_tmp=copy.copy(data_old)
        data_tmp.edge_index=[data_old.edge_index,data_new.edge_index]

        out = model(data_tmp.x, data_tmp.edge_index, data_tmp.batch)  # Perform a single forward pass.
        loss = criterion(out, data_tmp.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def train(model,train_loader,optimizer,criterion):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        #model.conv1.register_forward_hook(get_activation('conv3'))
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test_hybrid(model,loader):
    model.eval()
    correct = 0
    for data_old, data_new in zip(loader[0],loader[1]):
        data_tmp=copy.copy(data_old)
        data_tmp.edge_index=[data_old.edge_index,data_new.edge_index]
        out = model(data_tmp.x, data_tmp.edge_index, data_tmp.batch)  # Perform a single forward pass.
        pred = out.argmax(dim=1)
        correct += int((pred == data_tmp.y).sum())  # Check against ground-truth labels.
    return correct / len(loader[0].dataset)

def test(model,loader):
    model.eval()
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.

def predict_hybrid(model,loader):
    model.eval()
    pred=[]
    for data_old, data_new in zip(loader[0],loader[1]):
        data_tmp=copy.copy(data_old)
        data_tmp.edge_index=[data_old.edge_index,data_new.edge_index]
        out = model(data_tmp.x, data_tmp.edge_index, data_tmp.batch)  # Perform a single forward pass.
        pred.append(out.argmax(dim=1).tolist())
    return pred

def predict(model,loader):
    model.eval()
    pred=[]
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)  
        pred.append(out.argmax(dim=1).tolist())  # Use the class with highest probability.
          # Check against ground-truth labels.
    return pred  # Derive ratio of correct predictions.

def loss_hybrid(model,loader,criterion):
    model.eval()
    loss=0.
    for data_old, data_new in zip(loader[0], loader[1]):
        data_tmp=copy.copy(data_old)
        data_tmp.edge_index=[data_old.edge_index,data_new.edge_index]
        out = model(data_tmp.x, data_tmp.edge_index, data_tmp.batch)  # Perform a single forward pass.
        loss = criterion(out, data_tmp.y).sum()

    return loss/len(loader[0].dataset)

def loss(model,loader,criterion):
    model.eval()
    loss=0.
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        loss += criterion(out, data.y).sum()
    return loss/len(loader.dataset)

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
    type_input=input[5]
    if type_input=='AlphaFold':
        path=str(pdb_path)+'/'+'AF-'+str(my_protein)+'-F1-model_v3.pdb'
    if type_input=='PDB':
        path=str(pdb_path)+'/'+str(my_protein)+'.pdb'
    else:
        print('Unkown input type, must be AlphaFold or PDB')
        return 
    if exists(path):
        try:
            pdb_df=pdb2df(path)
            res2node=PDB2Graph.residueID2nodeID(pdb_df)
            #G=read_pdb(str(pdb_path)+'/'+str(my_protein)+'.pdb')
            if type_input=='AlphaFold':
                G=read_pdb(str(pdb_path)+'/'+'AF-'+str(my_protein)+'-F1-model_v3.pdb')
            if type_input=='PDB':
                G=read_pdb(str(pdb_path)+'/'+str(my_protein)+'.pdb')
            else:
                print('Unkown input type, must be AlphaFold or PDB')
                return
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

def clustering_graph(dataset,proteins):
    predetermined_cluster=True
    if predetermined_cluster==True:
        cluster_map={}
        with open('data_base/dataAll_cluster.dat', "r") as file:
            content = file.read()
        for line in content.splitlines():
            line=list(line.split(":"))
            cluster_map[line[0]]=line[1]

    clustered_dataset=[]
    old=[]
    new=[]
    t0 = time.time()
    for i,G in enumerate(dataset):
        G_tmp=copy.copy(G)

        if predetermined_cluster==True:
            no_existing_map=0
            my_cluster=[]
            protein_idx=G['prot_idx']
            prot_id=proteins[protein_idx]
            if prot_id in cluster_map and cluster_map[prot_id]!='None':
                my_cluster_tmp=cluster_map[prot_id]
                cluster_tmp=list(my_cluster_tmp.split(";"))
                for tmp in cluster_tmp:
                    group=[]
                    for member in tmp.split(","):
                        group.append(int(member))
                    my_cluster.append(group)
            else:
                no_existing_map+=1
                my_cluster=[list(range(len(G['x'])))]
        else:
            #my_cluster=GNN_clustering.spectral_cluster(G)
            my_cluster=GNN_clustering.modularity_clustering_simple(G)

        edges=G['edge_index'].numpy().T
        to_remove=[]
        for index,edge in enumerate(edges):
            found=0
            for c in my_cluster:
                if edge[0] in c and edge[1] in c:
                    found=1
                    break
            if found==0:
                to_remove.append(index)
        edges=np.delete(edges,obj=to_remove, axis=0)
        t = torch.from_numpy(edges.T)
        G_tmp['edge_index']=t
        new.append(G_tmp)
        old.append(G)
    t1 = time.time()
    clustered_dataset=new
    return clustered_dataset