import numpy as np
from proteingraph import read_pdb
from proteingraph.pin import pdb2df,compute_chain_pos_aa_mapping
from IPython.display import HTML
import pypdb 
from biopandas.pdb import PandasPdb
## proteingraph version "0.3.0"


#pdb_file = pypdb.get_pdb_file('2viu', filetype='cif', compression=False)


def number_of_nodes(pdb_id):
    G = read_pdb('pdb_library/'+str(pdb_id)+'.pdb')
    return G.number_of_nodes()

def number_of_edges(pdb_id):
    G = read_pdb('pdb_library/'+str(pdb_id)+'.pdb')
    return G.number_of_edges()

def edges(pdb_id):
    path='pdb_library/'+str(pdb_id)+'.pdb'
    G = read_pdb(path)
    res2node=residueID2nodeID(pdb_id)

    start_aa=[]
    target_aa=[]


    for edge in G.edges():
        residue_start=str(edge[0][:-3])
        residue_target=str(edge[1][:-3])
        node_start=res2node[residue_start]
        node_target=res2node[residue_target]
        start_aa.append(int(node_start))
        target_aa.append(int(node_target))

    result_edges=[start_aa,target_aa]
    return result_edges

def residueID2nodeID(pdb_id):
    path='pdb_library/'+str(pdb_id)+'.pdb'
    pdb_df=pdb2df(path)
    #atomic_df = PandasPdb().read_pdb(str(path)).df["ATOM"]
    chain_pos_aa=compute_chain_pos_aa_mapping(pdb_df)
    residue2node={}
    node_index=0

    for i in chain_pos_aa.keys():
        my_chain=chain_pos_aa[i]
        for item in my_chain:
            residue2node[str(i)+str(item)]=str(node_index)
            node_index+=1
    return residue2node

def nodes(pdb_id):
    path='pdb_library/'+str(pdb_id)+'.pdb'
    res2node=residueID2nodeID(pdb_id)
    G = read_pdb(path)
    sequence=['N']*G.number_of_nodes()
    for node in G.nodes():

        residueID=str(node[:-3])
        aa_name=str(node[-3:])
        nodeID=res2node[residueID]
        sequence[int(nodeID)]=aa_name

    return sequence