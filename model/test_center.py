import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import math, random, sys
from argparse import ArgumentParser
from collections import deque
from chemutils import remove_atommap, is_sim

import rdkit
import rdkit.Chem as Chem
import numpy as np
from rdkit.Chem import Descriptors
from concurrent import futures
from mol_tree import MolTree
from molcenter import MolCenter
from vocab import Vocab, common_atom_vocab
from datautils import MolTreeFolder

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_tree(smiles):
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    mol_tree = MolTree(smiles)
    return mol_tree

    
parser = ArgumentParser()
parser.add_argument("-t", "--test", dest="test_path")
parser.add_argument("-m", "--model", dest="model_path")
parser.add_argument("-d", "--save_dir", dest="save_dir")
parser.add_argument("-o", "--output", dest="output")
parser.add_argument("-st", "--start", type=int, dest="start", default=0)
parser.add_argument("-si", "--size", type=int, dest="size", default=0)

parser.add_argument("--vocab", type=str, default="../data/vocab.txt")
parser.add_argument('--knum', type=int, default=10)
parser.add_argument("--ncpu", type=int, default=10)
parser.add_argument("--seed", type=int, default=2021)
parser.add_argument('--hidden_size', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--embed_size', type=int, default=32)
parser.add_argument('--latent_size', type=int, default=32)
parser.add_argument('--depthG', type=int, default=5)
parser.add_argument('--depthT', type=int, default=3)

parser.add_argument('--use_atomic', action="store_false")
parser.add_argument('--sum_pool', action="store_false")
parser.add_argument('--use_atom_product', action="store_true")
parser.add_argument('--use_node_embed', action="store_true")
parser.add_argument('--use_brics', action="store_true")
parser.add_argument('--update_embed', action="store_true")
parser.add_argument('--use_attachatom', action="store_true")
parser.add_argument('--use_tree', action="store_true")
parser.add_argument('--use_feature', action="store_false")    
parser.add_argument('--use_product', action="store_false")    
parser.add_argument('--use_class', action="store_true")
parser.add_argument('--use_mess', action="store_true")
parser.add_argument('--use_latent_attachatom', action='store_true')
parser.add_argument('--network_type', type=str, default="gcn")

args = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
vocab = Vocab(vocab)
avocab = common_atom_vocab

model = MolCenter(vocab, common_atom_vocab, args)
try:
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device(device)))
except:
    raise ValueError("model does not exist")


if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

output_path = args.save_dir + args.output

if "-" in args.model_path: output_path += ("_" + args.model_path.split("-")[1])

res_file = open("%s_center_result.txt" % (output_path), 'w')
error_file = open("%s_center_error.txt" % (output_path), 'w')

data = []
with open(args.test_path) as f:
    for line in f.readlines()[1:]:
        s = line.strip("\r\n ").split(",")
        smiles = s[2].split(">>")
        data.append((int(s[0]), s[1], smiles[1], smiles[0]))

output = []
time1 = time.time()
start = int(args.start)
end = int(args.size) + start if args.size > 0 else len(data)

loader = MolTreeFolder(data[start:end], vocab, avocab, args.batch_size, use_atomic=args.use_atomic, \
                       use_class=args.use_class, use_brics=args.use_brics, del_center=False, use_feature=args.use_feature)

top_5_bool = np.zeros((len(loader.prod_list), 10))


bond_5_bool = []
atom_5_bool = []
lists = []
num = 0
iter_num = 0
# optimize input molecules

f = open("%s_center.out" % (output_path), 'w')
for batch in loader:
    classes, product_batch, product_tree, react_smiles, product_smiles, synthon_smiles, skip_idxs = batch
    
    with torch.no_grad():
        top_5_acc, bond_center_acc, atom_center_acc = model.validate_centers(classes, product_batch, product_tree, [], knum=args.knum)
    print("cur accuracy: %.4f" % (np.sum(top_5_acc[:, 0]== 1) / top_5_acc.shape[0]))
    acc_idx = 0
    strs = ""
    bond_5_bool.append(bond_center_acc)
    atom_5_bool.append(atom_center_acc)
    
    for i, (idx, smile) in enumerate(product_smiles):
        lists.append(idx)
        
        if i in skip_idxs:
            strs += ("%s %s %s\n" % (idx, smile, " ".join([str(j) for j in top_5_bool[num+i, :]])))
            continue
        
        top_5_bool[num + i, :] = top_5_acc[acc_idx, :]
        strs += ("%s %s %s\n" % (idx, smile, " ".join([str(j) for j in top_5_bool[num+i, :]])))
        acc_idx += 1
    
    f.write(strs)
    f.flush()
    iter_num += 1
    
    top_5_acc = np.sum(top_5_bool[:num+len(product_smiles), :], axis=0) / (num + len(product_smiles))
    num += len(product_smiles)
    
    bond_5_accs = np.concatenate( bond_5_bool, axis=0)
    bond_5_acc = np.sum(bond_5_accs, axis=0) / bond_5_accs.shape[0]
    
    atom_5_accs = np.concatenate( atom_5_bool, axis=0)
    atom_5_acc = np.sum(atom_5_accs, axis=0) / atom_5_accs.shape[0]
    print("cur: top 5 accuracy: %.4f  %.4f  %.4f  %.4f  %.4f" % (top_5_acc[0], top_5_acc[1], top_5_acc[2], top_5_acc[3], top_5_acc[4]))
    print("bond top 5[%d]: %.4f %.4f %.4f %.4f %.4f" % (bond_5_accs.shape[0], bond_5_acc[0], bond_5_acc[1], bond_5_acc[2], bond_5_acc[3], bond_5_acc[4]))
    print("atom top 5[%d]: %.4f %.4f %.4f %.4f %.4f" % (atom_5_accs.shape[0], atom_5_acc[0], atom_5_acc[1], atom_5_acc[2], atom_5_acc[3], atom_5_acc[4]))
    
    sys.stdout.flush()

f.close()
top_5_bool = np.sum(top_5_bool, axis=0) / len(loader.prod_list)
print("top 5 accuracy: %.4f  %.4f  %.4f  %.4f  %.4f" % (top_5_bool[0], top_5_bool[1], top_5_bool[2], top_5_bool[3], top_5_bool[4]))
