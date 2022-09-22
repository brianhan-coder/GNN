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
        value=[]
        with open(filename, "r") as feature_file:
            file_content = feature_file.read()
        for file_line in file_content.splitlines():
            line=np.array(list(file_line.split(" ")))
            value.append(float(line[1]))
            item=[line[0],line[1]]
            feature_list.append(item)

        min_value=min(value)
        max_value=max(value)
        for item in feature_list:
	        item[1]=str((float(item[1])-min_value)/(max_value-min_value))
        self.features[feature_name]=Convert(feature_list)
    
    def buildProtein(self,sequence):
        result = []
        AminoAcid = namedtuple("AminoAcid", self.features.keys())
        for aminoAcid in sequence:
            tmp={str(feature):str(feature_dict[aminoAcid]) for feature, feature_dict in self.features.items()}
            acid=AminoAcid(**tmp)
            result.append(acid)
        return result




