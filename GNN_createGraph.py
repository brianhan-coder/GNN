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


#构建命令行指令
parser = argparse.ArgumentParser(description="create graph library for GNN training")#构建指令模块
parser.add_argument('-d','--dataset', required=True, help='the protein dataset')#require必须输入，输入类型未指定
parser.add_argument('--pdb_path', required=True, help='path to the pdb files')
parser.add_argument('-g','--graph_path', required=True, help='path to the location where the graphs should be saved')
parser.add_argument('-i','--type_input', required=True, help='specify whether it is AlphaFold input or PDB input')
args = parser.parse_args()#传入参数
protein_dataset=args.dataset#参数赋值
pdb_path=args.pdb_path
graph_path=args.graph_path
type_input=args.type_input

### loading features

featureData = feature_embedding.FeatureData()
featureData.readFeatureFile("AA_features/AA_polarity.dat","polarity")
featureData.readFeatureFile("AA_features/AA_hydrophobicity.dat","hydrophobicity")
featureData.readFeatureFile("AA_features/AA_flexibility.dat","flexibility")
featureData.readFeatureFile("AA_features/AA_IDP_Scale.dat","IDP_scale")
featureData.readFeatureFile("AA_features/AA_charge.dat","charge")
featureData.readFeatureFile("AA_features/AA_PiSite.dat","PiSite")

#featureData.readFeatureFile("AA_features/AA_AlphaHelix.dat","AlphaHelix")
#featureData.readFeatureFile("AA_features/AA_AntiParallelBetaStrand.dat","AntiParallelBetaStrand")
#featureData.readFeatureFile("AA_features/AA_BetaSheet.dat","BetaSheet")
#featureData.readFeatureFile("AA_features/AA_BetaTurn.dat","BetaTurn")
#featureData.readFeatureFile("AA_features/AA_Bulkiness.dat","Bulkiness")
#featureData.readFeatureFile("AA_features/AA_BuriedResidues.dat","BuriedResidues")
#featureData.readFeatureFile("AA_features/AA_CoilParameter.dat","CoilParameter")
#featureData.readFeatureFile("AA_features/AA_GraphShapeIndex.dat","GraphShapeIndex")
#featureData.readFeatureFile("AA_features/AA_MolecularMass.dat","MolecularMass")
#featureData.readFeatureFile("AA_features/AA_NCondonsCoding.dat","NCondonsCoding")
#featureData.readFeatureFile("AA_features/AA_NHydrogenBondDonors.dat","NHydrogenBondDonors")
#featureData.readFeatureFile("AA_features/AA_NNonbondingOrbitals.dat","NNonbondingOrbitals")
#featureData.readFeatureFile("AA_features/AA_ParallelBetaStrand.dat","ParallelBetaStrand")
#featureData.readFeatureFile("AA_features/AA_RatioHeteroEndSide.dat","RatioHeteroEndSide")
#featureData.readFeatureFile("AA_features/AA_RecognitionFactors.dat","RecognitionFactors")
#featureData.readFeatureFile("AA_features/AA_Refractivity.dat","Refractivity")
#featureData.readFeatureFile("AA_features/AA_RelativeMutability.dat","RelativeMutability")
#featureData.readFeatureFile("AA_features/AA_accessibility.dat","accessibility")
#featureData.readFeatureFile("AA_features/AA_polarizability.dat","polarizability")
#featureData.readFeatureFile("AA_features/AA_upsilon_steric_parameter.dat","upsilon_steric_parameter")
#featureData.readFeatureFile("AA_features/AA_vanderWaalsVolume.dat","vanderWaalsVolume")
### load proteins

proteins=[]
graph_labels=[]
with open(protein_dataset, "r") as file: #r指定读文件，with as结构防止返回错误，建立一个临时指令空间
    content = file.read()
for line in content.splitlines():#取文件中的行
    line=np.array(list(line.split("\t")))#将行内信息以空格隔开再整入列表，之后构建array
    proteins.append(line[0])#将array中的第0项存入proteins
    graph_labels.append(int(line[1]))#将array中的第1项作为labels

tmp = list(zip(proteins, graph_labels))#compress proteins and graph_labels，构成一一对应的列表
random.shuffle(tmp)#随机化列表
proteins, graph_labels = zip(*tmp)#展开zip
proteins, graph_labels = list(proteins), list(graph_labels)#将array转回list


if __name__ == '__main__':
    ### parallel converting PDB to graphs
    for protein_index,my_protein in enumerate(proteins):#enumerate()将一个可遍历iterable数据对象组合为一个索引序列，同时列出数据和数据下标。
        input_list=[pdb_path,my_protein,featureData,graph_labels,protein_index,type_input]

        G=GNN_core.convert_pdb2graph(input_list)
        if G!=None:
            nx.write_gpickle(G, graph_path+'/'+str(my_protein)+".nx")
