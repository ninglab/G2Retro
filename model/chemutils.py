# Modified from https://github.com/wengong-jin/iclr19-graph2graph
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import BRICS
from rdkit import DataStructs
from rdkit.Chem import AllChem
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from vocab import Vocab
import torch
import pdb

MST_MAX_WEIGHT = 100 
MAX_NCAND = 100
BOND_LIST = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    

def canonicalize(smiles, add_atom_num=False):
    try:        
        tmp = Chem.MolFromSmiles(smiles)
    except Exception as e:
        print(e)
        return smiles
    
    if tmp is None:
        print('wrong smiles: %s' % (smiles))
        return smiles
    
    tmp = Chem.RemoveHs(tmp)
    [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
    smiles = Chem.MolToSmiles(tmp)
    
    if add_atom_num:
        mol = Chem.MolFromSmiles(smiles)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
        smiles = Chem.MolToSmiles(mol)
        return smiles
    else:
        return smiles
    
def get_mol(smiles, sanitize=True):
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is None: 
        return None
    Chem.Kekulize(mol)
    return mol

def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


def sanitize(mol):
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception:
        return None
    return mol

def set_atommap(mol, num=0):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)

def get_uniq_atoms(graphs, cliques, attach_atoms, label, avocab):
    local_dict = []
    adj_mat = [{} for _ in cliques]
    aidxs = []
    pdb.set_trace()
    for i, atom in enumerate(cliques):
        aidxs.append(avocab[graphs.nodes[atom]['label']])
        edges = [edge[1] for edge in graphs.edges(atom) if edge[1] in cliques]
        adj_mat[i][1] = []
        for atom2 in edges:
            adj_mat[i][1].append((avocab[graphs.nodes[atom2]['label']], graphs[atom][atom2]['label']))
        
        adj_mat[i][1].sort()
    
    unmatched_idxs = [i for i in range(len(cliques))]
    matched_idxs = {}
    visited_idxs = []
    while len(unmatched_idx) > 0:
        for i, idx in enumerate(unmatched_idxs[:-1]):
            for jdx in unmatched_idxs[i+1:]:
                if adj_mat[idx] == adj_mat[jdx]:
                    if idx not in matched_idxs:
                        matched_idxs[idx] = [jdx]
                    else:
                        matched_idxs[idx].append(jdx)
        
                    if idx not in visited_idxs:
                        visited_idxs.append(idx)
                    
                    visited_idxs.append(jdx)
        
    return None

def bond_equal(bond1, bond2):
    begin_atom1 = bond1.GetBeginAtom()
    end_atom1 = bond1.GetEndAtom()
    bond_val1 = int(bond1.GetBondTypeAsDouble())
    
    begin_atom2 = bond2.GetBeginAtom()
    end_atom2 = bond2.GetEndAtom()
    bond_val2 = int(bond2.GetBondTypeAsDouble())
    if bond_val1 != bond_val2: return 0
    if atom_equal(begin_atom1, begin_atom2) and atom_equal(end_atom1, end_atom2):
        return 1
    elif atom_equal(begin_atom1, end_atom2) and atom_equal(begin_atom2, end_atom1):
        return 2

def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol

def remove_atommap(smiles):
    mol = get_mol(smiles)
    set_atommap(mol)
    return get_smiles(mol)

def get_idx_from_mapnum(mol):
    map_dict = {}
    for atom in mol.GetAtoms():
        atom_num = atom.GetAtomMapNum()
        if atom_num > 0:
            map_dict[atom_num] = atom.GetIdx()
    return map_dict

def get_mapnum_from_idx(mol):
    map_dict = {}
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        map_dict[atom_idx] = atom.GetAtomMapNum()
    return map_dict
    
def get_clique_mol(mol, atoms):
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol) #We assume this is not None
    return new_mol


def brics_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1: #special case
        return [[0]], []

    cliques = []
    atom_clique_dict = {}
    
    mapnum_to_idx = get_idx_from_mapnum(mol)
    if len(mapnum_to_idx) == 0:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
            mapnum_to_idx[atom.GetIdx()] = atom.GetIdx()
    
    frag_smiles = Chem.MolToSmiles(BRICS.BreakBRICSBonds(mol))
    for frag_smile in frag_smiles.split("."):
        frag_mol = Chem.MolFromSmiles(frag_smile)
        clique = []
        for atom in frag_mol.GetAtoms():
            if atom.GetSymbol() == "*": continue
            idx = mapnum_to_idx[atom.GetAtomMapNum()]
            clique.append( idx )
            atom_clique_dict[ idx ] = len(cliques)
        cliques.append(clique)
    
    edges = []
    for bond in mol.GetBonds():
        begin_atom_idx = bond.GetBeginAtom().GetIdx()
        end_atom_idx = bond.GetEndAtom().GetIdx()
        if atom_clique_dict[begin_atom_idx] == atom_clique_dict[end_atom_idx]: continue
        edges.append( (atom_clique_dict[begin_atom_idx], atom_clique_dict[end_atom_idx], begin_atom_idx, end_atom_idx) )
        
    return (cliques, edges)


# The function below is Modified from https://github.com/wengong-jin/iclr19-graph2graph/blob/master/fast_jtnn/chemutils.py
"""
iclr19-graph2graph
Copyright (c) 2019 Wengong Jin
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
def tree_decomp(mol, decompose_ring=False):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1: #special case
        return [[0]], []

    cliques = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            cliques.append([a1,a2])

    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques.extend(ssr)

    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)
    
    inter_num_thd = 2 if not decompose_ring else 3
    #Merge Rings with intersection >= 2 atoms
    for i in range(len(cliques)):
        if len(cliques[i]) <= 2: continue
        for atom in cliques[i]:
            for j in nei_list[atom]:
                if i >= j or len(cliques[j]) <= 2: continue
                inter = set(cliques[i]) & set(cliques[j])
                
                if len(inter) >= inter_num_thd:
                    cliques[i].extend(cliques[j])
                    cliques[i] = list(set(cliques[i]))
                    cliques[j] = []
   
    
    cliques = [c for c in cliques if len(c) > 0]
    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)
    
    #Build edges and add singleton cliques
    edges = defaultdict(int)
    for atom in range(n_atoms):
        if len(nei_list[atom]) <= 1: 
            continue
        cnei = nei_list[atom]
        bonds = [c for c in cnei if len(cliques[c]) == 2]
        rings = [c for c in cnei if len(cliques[c]) > 2]
        if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2): #In general, if len(cnei) >= 3, a singleton should be added, but 1 bond + 2 ring is currently not dealt with.
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1,c2)] = 1
        elif len(rings) > 2: #Multiple (n>2) complex rings
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1,c2)] = MST_MAX_WEIGHT - 1 # Must be selected in the tree
        else:
            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1,c2 = cnei[i],cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    if edges[(c1,c2)] < len(inter):
                        edges[(c1,c2)] = len(inter) #cnei[i] < cnei[j] by construction
                        
    edges = [(u[0],u[1],MST_MAX_WEIGHT-v) for u,v in edges.items()]
    if len(edges) == 0:
        return cliques, edges

    
    #Compute Maximum Spanning Tree
    row,col,data = zip(*edges)
    n_clique = len(cliques)
    clique_graph = csr_matrix( (data,(row,col)), shape=(n_clique,n_clique) )
    junc_tree = minimum_spanning_tree(clique_graph)
    row,col = junc_tree.nonzero()
    edges = [(row[i],col[i]) for i in range(len(row))]
    return (cliques, edges)

def atom_equal(a1, a2):
    return a1.GetSymbol() == a2.GetSymbol() and a1.GetFormalCharge() == a2.GetFormalCharge()

def get_ranked_atom_charges(atom_charge_logits):
    dicts = []
    charge_size = atom_charge_logits.shape[1]
    atom_size = atom_charge_logits.shape[0]
    
    dicts = [(tuple(), 0)]
    for i in range(atom_size):
        new_dicts = []
        for charge, lh in dicts:
            for ich in range(charge_size):
                new_dicts.append( (charge + (ich, ), lh + atom_charge_logits[i][ich].item()) )
        dicts = new_dicts
    
    dicts = sorted(dicts, key=lambda x: x[1], reverse=True)
    return dicts

def check_atom_valence(atom):
    """ Check whether the atom can connect with new atoms or not
    """
    max_valence = max([x for x in Chem.GetPeriodicTable().GetValenceList(atom.GetAtomicNum())])
    if atom.GetSymbol() == 'P': max_valence = 5
    
    if atom.GetTotalNumHs() > 0 or atom.GetFormalCharge() > 0 or atom.GetTotalValence() <= max_valence:
        return True
    else:
        return False

def check_attach_atom_valence(atom1, atom2):
    """ Check whether the atom1 can attach with atom2 without violating valency rules
    """
    max_valence = max([x for x in Chem.GetPeriodicTable().GetValenceList(atom1.GetAtomicNum())])
    if atom1.GetSymbol() == 'P':
        max_valence = 5
    
    ocupy_val1 = atom1.GetTotalValence() - atom1.GetTotalNumHs()
    ocupy_val2 = atom2.GetTotalValence() - atom2.GetTotalNumHs()
    
    if ocupy_val1 + ocupy_val2 <= max_valence + atom1.GetFormalCharge():
        return True
    else:
        return False

def graph_to_mol(graph, keep_atom=True, return_map=False, debug=False):
    """ convert the networkx graph back to molecule
        Arguments:
            keep_atom: whether keeping the new atoms in graph or not;
                       when keep_atom=False, the output molecule is a partial molecule with some new atoms removed; 
                       and this option is used to generate partial molecule from full molecule during training)
            return_map: whether returning the atom mapped dictionary to align the original atom index and the atom index in the output canonical molecules
            debug: print more information for debuging
    """
    def safe_remove(idx):
        for edge in graph.edges(idx):
            if "revise" not in graph[edge[0]][edge[1]] or graph[edge[0]][edge[1]]['revise'] != 1:
                return False
        return True
    
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    map_dict = {}
    jdx = 0
    for idx in graph.nodes:
        atom_label = graph.nodes[idx]['label']
        if not keep_atom and graph.nodes[idx]['idx'] == 0:
            if safe_remove(idx): continue
        
        map_dict[idx] = jdx
        jdx += 1
        atom = Chem.Atom(atom_label)
        
        if 'new_charge' in graph.nodes[idx]:
            try:
                atom.SetFormalCharge(graph.nodes[idx]['new_charge'])
            except:
                pdb.set_trace()
        else:
            atom.SetFormalCharge(graph.nodes[idx]['charge'])
        
        if 'idx' in graph.nodes[idx]: atom.SetAtomMapNum(graph.nodes[idx]['idx'])
        new_mol.AddAtom(atom)
        if debug:
            print("add idx %d num of atom %d" % (idx, new_mol.GetNumAtoms()))
        
    for edge in graph.edges:
        if edge[0] not in map_dict or edge[1] not in map_dict: continue
        
        beginatom_idx = map_dict[edge[0]]
        endatom_idx = map_dict[edge[1]]
        
        if new_mol.GetBondBetweenAtoms(beginatom_idx, endatom_idx) is not None: continue
        
        bond_label = graph[edge[0]][edge[1]]['label']
        
        atom1 = new_mol.GetAtomWithIdx(beginatom_idx)
        atom2 = new_mol.GetAtomWithIdx(endatom_idx)
        
        if 'new_label' in graph[edge[0]][edge[1]]:
            
            new_label = graph[edge[0]][edge[1]]['new_label']
            if new_label is None: continue
            new_mol.AddBond(beginatom_idx, endatom_idx, BOND_LIST[new_label])
        elif "revise" in graph[edge[0]][edge[1]] and graph[edge[0]][edge[1]]['revise'] == 1:
            continue
        else:
            new_mol.AddBond(beginatom_idx, endatom_idx, BOND_LIST[bond_label])
    
    
    mol = new_mol.GetMol()
    mol.UpdatePropertyCache()
    if return_map:
        smile = Chem.MolToSmiles(mol)
        atom_orders = [int(num) for num in mol.GetProp('_smilesAtomOutputOrder')[1:-1].split(",") if len(num) > 0]
        if debug:
            print("len order %d" % (len(atom_orders)))
            print(map_dict)
            print(atom_orders)
            print(mol.GetProp('_smilesAtomOutputOrder'))
            
        for key in map_dict:
            map_dict[key] = atom_orders.index(map_dict[key])
        mol = get_mol(smile)
        if debug:
            print(smile)
            print(Chem.MolToSmiles(mol))
            print("atom num %d" % (mol.GetNumAtoms()))
        return mol, map_dict
    else:
        Chem.SanitizeMol(mol)
        Chem.Kekulize(mol)
        return mol


#get synthon molecules from reactant smiles
def get_synthon_from_smiles(smiles):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    old_mol = get_mol(smiles)
    map_dict = {}
    jdx = 0
    for atom in old_mol.GetAtoms():
        if atom.GetAtomMapNum() == 0: continue
        map_dict[atom.GetIdx()] = jdx
        jdx += 1
        newatom = Chem.Atom(atom.GetSymbol())
        newatom.SetFormalCharge(atom.GetFormalCharge())
        newatom.SetAtomMapNum(atom.GetAtomMapNum())
        new_mol.AddAtom(newatom)
    
    for bond in old_mol.GetBonds():
        begin_atom = bond.GetBeginAtom()
        if begin_atom.GetAtomMapNum() == 0: continue

        end_atom = bond.GetEndAtom()
        if end_atom.GetAtomMapNum() == 0: continue
        
        beginatom_idx = map_dict[begin_atom.GetIdx()]
        endatom_idx = map_dict[end_atom.GetIdx()]
        
        new_mol.AddBond(beginatom_idx, endatom_idx, bond.GetBondType())
        
    mol = new_mol.GetMol()
    mol.UpdatePropertyCache() 
    
    smiles = Chem.MolToSmiles(mol)
    return mol, smiles
    
    
def mol_to_graph(mol):
    """ build networkx graph from molecule
    """
    graph = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(mol))
    for atom in mol.GetAtoms():
        graph.nodes[atom.GetIdx()]['label'] = (atom.GetSymbol(), atom.GetFormalCharge())

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        btype = BOND_LIST.index( bond.GetBondType() )
        graph[a1][a2]['label'] = btype
        graph[a2][a1]['label'] = btype

    return graph

def attach_mol_graph(graph, mol, atom1_idxs, atom2_idxs):
    """ attach the atom2 in the fragment represented by "mol" to atom1 in molecular graph represented by "graph"
    """
    num_atoms = mol.GetNumAtoms()
    amap = {}
    for idx in range(num_atoms):
        if idx in atom2_idxs:
            amap[idx] = atom1_idxs[atom2_idxs.index(idx)]
            continue
        atom = mol.GetAtomWithIdx(idx)
        atom_idx = len(graph)
         
        graph.add_node(atom_idx)
        graph.nodes[atom_idx]['label'] = atom.GetSymbol()
        graph.nodes[atom_idx]['charge'] = atom.GetFormalCharge()
        graph.nodes[atom_idx]['valence'] = atom.GetTotalValence()
        graph.nodes[atom_idx]['aroma'] = atom.GetIsAromatic()
        graph.nodes[atom_idx]['num_h'] = atom.GetTotalNumHs()
        graph.nodes[atom_idx]['atomic_num'] = atom.GetAtomicNum()
        graph.nodes[atom_idx]['in_ring'] = atom.IsInRing()
        
        graph.nodes[atom_idx]['bonds'] = []
        graph.nodes[atom_idx]['rings'] = []
        amap[idx] = atom_idx        
    
    new_mol = None
    if mol.GetAtomWithIdx(0).GetIsAromatic():
        new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        Chem.Kekulize(new_mol)
    
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        btype = BOND_LIST.index( bond.GetBondType() )
        if end_idx not in atom2_idxs or begin_idx not in atom2_idxs:
            map_begin_idx = amap[begin_idx]
            map_end_idx = amap[end_idx]
            graph.add_edge(map_begin_idx, map_end_idx)
            graph.add_edge(map_end_idx, map_begin_idx)
            graph[map_begin_idx][map_end_idx]['label'] = btype
            graph[map_begin_idx][map_end_idx]['dir'] = 0
            graph[map_begin_idx][map_end_idx]['mess_idx'] = len(graph.edges) - 1
            
            graph[map_begin_idx][map_end_idx]['is_conju'] = graph[map_end_idx][map_begin_idx]['is_conju'] = bond.GetIsConjugated()
            graph[map_begin_idx][map_end_idx]['is_aroma'] = graph[map_end_idx][map_begin_idx]['is_aroma'] = bond.GetIsAromatic() 
            graph[map_begin_idx][map_end_idx]['in_ring'] = graph[map_end_idx][map_begin_idx]['in_ring'] = bond.IsInRing()
            
            graph[map_end_idx][map_begin_idx]['label'] = btype
            graph[map_end_idx][map_begin_idx]['dir'] = 1
            graph[map_end_idx][map_begin_idx]['mess_idx'] = len(graph.edges)

    return graph
    
def get_bonds_atommap(mol, mark=0):
    """ return all the edges with the atoms associated with the atom-map numbers
    """
    bonds_with_atommap = {}
    
    bonds_without_atommap = {}
    
    for bond in mol.GetBonds():
        begin_atom_num = bond.GetBeginAtom().GetAtomMapNum()
        end_atom_num   = bond.GetEndAtom().GetAtomMapNum()
        
        begin_atom_idx = bond.GetBeginAtom().GetIdx()
        end_atom_idx   = bond.GetEndAtom().GetIdx()
        
        bond_type = (int(bond.GetBondTypeAsDouble()), int(bond.GetIsAromatic()))
        
        if begin_atom_num != 0 and end_atom_num != 0:
            bonds_with_atommap[(begin_atom_num, end_atom_num)] = (begin_atom_idx, end_atom_idx, bond_type, mark)
            bonds_with_atommap[(end_atom_num, begin_atom_num)] = (end_atom_idx, begin_atom_idx, bond_type, mark)
        else:
            bonds_without_atommap[(mark, begin_atom_idx, end_atom_idx)] = bond_type[0]
            bonds_without_atommap[(mark, end_atom_idx, begin_atom_idx)] = bond_type[0]
    
    return bonds_with_atommap, bonds_without_atommap
 
def copy_bond_dir(product, pre_react):
    """ copy the direction of bonds from the product molecule to the predicted reactant molecules
    """
    bond_dir_map = {}
    bond_stereo_map = {}
    
    for bond in product.GetBonds():
        begin_atom = bond.GetBeginAtom().GetAtomMapNum()
        end_atom = bond.GetEndAtom().GetAtomMapNum()
        if bond.GetBondDir() != Chem.rdchem.BondDir.NONE:
            bond_dir_map[(begin_atom, end_atom)] = bond.GetBondDir()
            
        if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE:
            bond_stereo_map[(begin_atom, end_atom)] = bond.GetStereo()
        
    change_mol = Chem.RWMol(pre_react)
    for bond in change_mol.GetBonds():
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        
        begin_atom_mapnum = begin_atom.GetAtomMapNum()
        end_atom_mapnum = end_atom.GetAtomMapNum()
        
        if begin_atom_mapnum == 0 or end_atom_mapnum == 0:
            continue
        
        if (end_atom_mapnum, begin_atom_mapnum) in bond_stereo_map:
            begin_atom_mapnum, end_atom_mapnum = end_atom_mapnum, begin_atom_mapnum
        
        if (begin_atom_mapnum, end_atom_mapnum) in bond_stereo_map:
            bond.SetStereo(bond_stereo_map[(begin_atom_mapnum, end_atom_mapnum)])
            
        if (end_atom_mapnum, begin_atom_mapnum) in bond_dir_map:
            begin_atom_mapnum, end_atom_mapnum = end_atom_mapnum, begin_atom_mapnum
        
        if (begin_atom_mapnum, end_atom_mapnum) in bond_dir_map:
            bond.SetBondDir(bond_dir_map[(begin_atom_mapnum, end_atom_mapnum)])
            
    return change_mol

from rdchiral.chiral import copy_chirality
from rdkit.Chem import SanitizeMol, SanitizeFlags
from rdkit.Chem.AllChem import AssignStereochemistry

def add_chirality(product, pred_react):
    """ copy the atom chirality and bond direction from the product molecule to the predicted reactant molecule
    """
    prod_mol = Chem.MolFromSmiles(product)
    react_mol = Chem.MolFromSmiles(pred_react)
    
    react_atom_map = {}
    
    for atom in react_mol.GetAtoms():
        mapnum = atom.GetAtomMapNum()
        react_atom_map[mapnum] = atom
        
    for atom in prod_mol.GetAtoms():
        mapnum = atom.GetAtomMapNum()
        
        ratom = react_atom_map[mapnum]
        
        copy_chirality(atom, ratom)
    
    chiral_react_smiles = Chem.MolToSmiles(react_mol, isomericSmiles=True)
    react_mol = Chem.MolFromSmiles(chiral_react_smiles)
    change_react_mol = copy_bond_dir(prod_mol, react_mol)
    
    SanitizeMol(change_react_mol, sanitizeOps=SanitizeFlags.SANITIZE_ALL, catchErrors=False)
    AssignStereochemistry(change_react_mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
        
    chiral_react_smiles = Chem.MolToSmiles(change_react_mol, isomericSmiles=True)
    
    return chiral_react_smiles

def is_sim(smile1, smile2):
    smile1 = canonicalize(smile1)
    smile2 = canonicalize(smile2)
      
    if smile1 == smile2: return True
    else: return False
