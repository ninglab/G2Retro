# Code from https://github.com/wengong-jin/iclr19-graph2graph
import rdkit.Chem as Chem
import copy
import os
import argparse
import pickle

def get_slots(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return [(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs()) for atom in mol.GetAtoms()]

class Vocab(object):
    def __init__(self, smiles_list):
        self.vocab = smiles_list
        self.vmap = {x:i for i,x in enumerate(self.vocab)}
        if len(smiles_list[0]) > 1:
            self.max_len = max([Chem.MolFromSmiles(smiles).GetNumAtoms() for smiles in smiles_list])
        
    def __getitem__(self, smiles):
        return self.vmap[smiles]

    def get_smiles(self, idx):
        return self.vocab[idx]

    def size(self):
        return len(self.vocab)

COMMON_ATOMS = ['B','C', 'N', 'O', 'F', 'Mg', 'Si', 'P', 'S', 'Cl', 'Cu', 'Zn', 'Se', 'Br', 'Sn', 'I']

common_atom_vocab = Vocab(COMMON_ATOMS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    vocabs = set()
    
    for f in os.listdir(args.path):
        if "tensors" in f and "pkl" in f:
            
            with open(args.path + f, 'rb') as tmp:
                mol_trees = pickle.load(tmp)
            
            for _, _, react_tree in mol_trees:
                for _, node_idx in react_tree.order:
                    if node_idx >= 0: vocabs.add(react_tree.mol_tree.nodes[node_idx]['label'])

            del mol_trees

    out_file = open(args.output, 'w')
    for word in vocabs:
        out_file.write("%s\n" % (word))
    out_file.close()
