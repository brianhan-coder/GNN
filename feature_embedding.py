import numpy as np
from collections import namedtuple

def Convert(lst):
    res_dct = {lst[i][0]: lst[i][1] for i in range(0, len(lst))}
    return res_dct

class FeatureData:
    def __init__(self):
        self.features = {}

    def readFeatureFile(self,filename,feature_name):
        feature_list=[]
        with open(filename, "r") as feature_file:
            file_content = feature_file.read()
        for file_line in file_content.splitlines():
            line=np.array(list(file_line.split(" ")))

            item=[line[0],line[1]]
            feature_list.append(item)
        self.features[feature_name]=Convert(feature_list)
    
    def buildProtein(self,*sequence):
        result = []
        AminoAcid = namedtuple("AminoAcid", self.features.keys())
        #print(self.features)
        for aminoAcid in sequence:
            tmp={feature:feature_dict[aminoAcid] for feature, feature_dict in self.features.items()}
            acid=AminoAcid(**tmp)
            result.append(acid)
        return result




