import numpy as np
from proteingraph import read_pdb
from proteingraph.pin import filter_dataframe,compute_chain_pos_aa_mapping,compute_rgroup_dataframe,compute_distmat,get_interacting_atoms
from IPython.display import HTML
from biopandas.pdb import PandasPdb
## proteingraph version "0.3.0"
import time


def number_of_nodes(pdb_path,pdb_id):
    G = read_pdb(str(pdb_path)+'/'+str(pdb_id)+'.pdb')
    return G.number_of_nodes()

def number_of_edges(pdb_path,pdb_id):
    G = read_pdb(str(pdb_path)+'/'+str(pdb_id)+'.pdb')
    return G.number_of_edges()

def edge(G,res2node):
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

def residueID2nodeID(pdb_df):
    chain_pos_aa=compute_chain_pos_aa_mapping(pdb_df)
    residue2node={}
    node_index=0

    for i in chain_pos_aa.keys():
        my_chain=chain_pos_aa[i]
        for item in my_chain:
            residue2node[str(i)+str(item)]=str(node_index)
            node_index+=1
    return residue2node

def node(G,res2node):
    sequence=['N']*G.number_of_nodes()
    for node in G.nodes():

        residueID=str(node[:-3])
        aa_name=str(node[-3:])
        nodeID=res2node[residueID]
        sequence[int(nodeID)]=aa_name
    return sequence


def edge_feature(G):
    edge_type=[]
    for _,_,d in G.edges(data=True):
        edge_type.append(d['kind'])
    return edge_type

def add_pi_pi_interactions(G,pdb_df):
    """
    Find all (non-aromatic) pi-pi interactions.
    Performs searches between the following residues:
    "TYR","PHE","TRP","HIS","GLN","ASN","GLU","ASP","ARG","GLY","SER","THR","PRO"
    Criteria: R-group residues are within 6A distance.
    """
    non_aromatic_pi_RESIS=["TYR","PHE","TRP","HIS","GLN","ASN","GLU","ASP","ARG","GLY","SER","THR","PRO"]
    rgroup_df = compute_rgroup_dataframe(pdb_df)
    non_aromatic_pi_df = filter_dataframe(rgroup_df, "residue_name", non_aromatic_pi_RESIS, True)
    distmat = compute_distmat(non_aromatic_pi_df)
    interacting_atoms = get_interacting_atoms(6, distmat)

    # this function is NOT heritaged from proteingraph.pin, its definition is below
    add_interacting_resis_additional(G, interacting_atoms, non_aromatic_pi_df, ["pipi"])

def add_interacting_resis_additional(G, interacting_atoms, dataframe, kind): 
    # modify this function from pin.py in order to add new bond types
    resi1 = dataframe.loc[interacting_atoms[0]]["node_id"].values
    resi2 = dataframe.loc[interacting_atoms[1]]["node_id"].values

    interacting_resis = set(list(zip(resi1, resi2)))
    for i1, i2 in interacting_resis:
        if i1 != i2:
            if G.has_edge(i1, i2):
                for k in kind:
                    if k not in G.edges[i1, i2]["kind"]:
                        G.edges[i1, i2]["kind"].append(k)
            else:
                G.add_edge(i1, i2, kind=list(kind))