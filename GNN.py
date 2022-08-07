import numpy as np

import feature_embedding

featureData = feature_embedding.FeatureData()
featureData.readFeatureFile("AA_features/AA_charge.dat","charge")
featureData.readFeatureFile("AA_features/AA_hydrophobic.dat","hydrophobicity")


for aminoAcid in featureData.buildProtein('F','A','B'):
    print(aminoAcid.charge)
    
