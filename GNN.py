import numpy as np
import torch
import feature_embedding
import PDB2Graph
#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

### loading features
featureData = feature_embedding.FeatureData()
featureData.readFeatureFile("AA_features/AA_charge.dat","charge")
featureData.readFeatureFile("AA_features/AA_hydrophobic.dat","hydrophobicity")

# loading PDB for the protein structure
nodes=PDB2Graph.nodes('2VIU')
edges=PDB2Graph.edges('2VIU')
print(nodes)


### readin feature list of all amino acids
complete_list_feature=[]
for aminoAcid in featureData.buildProtein('F','A','B'):
    my_features=[float(tmp) for tmp in list(aminoAcid)]
    complete_list_feature.append(my_features)

#nodes_feature=torch.tensor(complete_list_feature,dtype=torch.float) # feature vector of all nodes

#edge_index = torch.tensor([[0, 1, 2, 0, 3],[1, 0, 1, 3, 2]], dtype=torch.long) # edges, 1st list: index of the source nodes, 2nd list: index of target nodes.
