import networkx as nx
import feature_embedding
import PDB2Graph
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random 
import os
from sklearn.cluster import KMeans
import leidenalg as la
import igraph as ig
from itertools import groupby,chain
from operator import itemgetter

Q_thres=0.01
ave_node_thres=0.2
min_node_cluster=5
max_cluster_size=14

def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

def numberEdge_perNode(first,second,num_nodes):
    node_by_res=[0]*num_nodes
    for i in range(0,len(first)):
        node_by_res[first[i]]+=1
        node_by_res[second[i]]+=1
    return node_by_res

def contact_map(first,second,num_nodes):
    
    contact=np.zeros((num_nodes,num_nodes))
    for i in range(len(first)):
        a=first[i]
        b=second[i]
        contact[b][a]=1
        contact[a][b]=1
    return contact


def spectral_cluster(G):
    num_nodes=len(G['x'].numpy())
    first=G['edge_index'].numpy()[0]
    second=G['edge_index'].numpy()[1]

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
    x=range(len(vals))
    #clusters=range(10,50)
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
        if asy_degree<best_asy:
            best_cluster=contact_cluster
            best_asy=asy_degree
            best_n_cluster=n_cluster
            best_node_label=kmeans.labels_

    final_cluster_num=max(best_node_label)+1
    final_cluster=[[] for x in range(final_cluster_num)]

    for node in range(num_nodes):
        final_cluster[best_node_label[node]].append(node)
    return final_cluster

def find_dense_module(edge_per_node,m,part,A):
    Q_list=[]
    ave_node=[]
    cluster_size=[]
    cluster_index=[]
    for c in range(len(part)):
        my_cluster=part[c]
        num_edge_of_nodes=[0]*len(my_cluster)
        Q=0

        for my_v in my_cluster:
            for my_w in my_cluster:
                Q+=(A[my_v][my_w]-(edge_per_node[my_v]*edge_per_node[my_w])/(2*m))/(2*m)
                if A[my_v][my_w]==1:
                    num_edge_of_nodes[my_cluster.index(my_v)]+=1./len(my_cluster)
                cluster_size.append(len(my_cluster))
                cluster_index.append(c)
        Q_list.append(Q)
        ave_node.append(np.mean(num_edge_of_nodes))
        #print(Q,min(num_edge_of_nodes),len(my_cluster))

    ave_node = np.array(ave_node)[np.argsort(Q_list)]
    Q_list = np.array(Q_list)[np.argsort(Q_list)]
    cluster_index = np.array(cluster_index)[np.argsort(Q_list)]
    cluster_size=np.array(cluster_size)[np.argsort(Q_list)]

    if Q_list[-1]>Q_thres and ave_node[-1]>=ave_node_thres and cluster_size[-1]>min_node_cluster:
        return Q_list[-1],ave_node[-1],part[cluster_index[-1]]
    else:
        return Q_list[-1],ave_node[-1],None

def modularity_clustering(G):
    optimiser = la.Optimiser()
    num_nodes=len(G['x'].numpy())
    first=G['edge_index'].numpy()[0]
    second=G['edge_index'].numpy()[1]

    A=contact_map(first,second,num_nodes)
    G_ig = ig.Graph.Adjacency((A > 0).tolist())
    final_cluster=[]
    edge_per_node=numberEdge_perNode(first,second,num_nodes)
    m=len(first)

    max_comm_size_list=range(6,max_cluster_size)
    best_Q=1
    while best_Q>Q_thres:
        best_Q=0
        for max_comm_size in max_comm_size_list:
            part = la.find_partition(G_ig, la.ModularityVertexPartition,max_comm_size=max_comm_size)

            Q,ave_n_node,c=find_dense_module(edge_per_node,m,part,A)
            if Q>best_Q and c!=None:
                best_c=c
                best_Q=Q
                best_ave_n_node=ave_n_node
        for i in best_c:
            for j in best_c:
                if G_ig.are_connected(i, j):
                    G_ig.delete_edges([(i,j)])
        final_cluster.append(best_c)

    node_in_cluster = list(chain.from_iterable(final_cluster))
    remove_list=list(set(node_in_cluster))
    remaining_nodes= [i for i in range(num_nodes) if i not in remove_list]
    grouped_rem_nodes=ranges(remaining_nodes)
    singleton=[]
    for node in grouped_rem_nodes:
        if node[1]-node[0]>0:
            final_cluster.append(list(range(node[0],node[1]+1)))
        else:
            singleton.append(list(range(node[0],node[1]+1))[0])
    for single in singleton:
        bonds=[0]*len(final_cluster)
        for fc in final_cluster:
            for fc_node in fc:
                if A[single][fc_node]==1:
                    bonds[final_cluster.index(fc)]+=1
        final_cluster[np.argmax(bonds)].append(single)
    return final_cluster

def modularity_clustering_simple(G):
    num_nodes=len(G['x'].numpy())
    first=G['edge_index'].numpy()[0]
    second=G['edge_index'].numpy()[1]
    A=contact_map(first,second,num_nodes)
    G_ig = ig.Graph.Adjacency((A > 0).tolist())
    max_comm_size=16
    part = la.find_partition(G_ig, la.ModularityVertexPartition,max_comm_size=max_comm_size)
    cluster=[]
    for c in range(len(part)):
        my_cluster=part[c]
        cluster.append(my_cluster)

    return cluster