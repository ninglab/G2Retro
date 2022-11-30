import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import math, random, sys
from argparse import ArgumentParser
from collections import deque
from chemutils import is_sim

import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors
from concurrent import futures
from mol_tree import MolTree
from molcenter import MolCenter
from molsynthon import MolSynthon
from vocab import Vocab, common_atom_vocab
from datautils import MolTreeFolder
import numpy as np

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_tree(smiles):
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    mol_tree = MolTree(smiles)
    return mol_tree

parser = ArgumentParser()

parser.add_argument("-t", "--test", dest="test_path")
parser.add_argument("-m1", "--mod_cen_path", dest="mod_cen_path")
parser.add_argument("-m2", "--mod_syn_path", dest="mod_syn_path")
parser.add_argument("-d", "--save_dir", dest="save_dir")
parser.add_argument("-o", "--output", dest="output")
parser.add_argument("-st", "--start", type=int, dest="start", default=0)
parser.add_argument("-si", "--size", type=int, dest="size", default=0)

parser.add_argument("--vocab", type=str, default="../data/vocab.txt")
parser.add_argument("--ncpu", type=int, default=10)
parser.add_argument("--decode_type", type=int, default=2)
parser.add_argument("--seed", type=int, default=2021)
parser.add_argument("--knum", type=int, default=10)

parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--hidden_sizeS', type=int, default=32)
parser.add_argument('--hidden_sizeC', type=int, default=32)
parser.add_argument('--embed_sizeS', type=int, default=32)
parser.add_argument('--embed_sizeC', type=int, default=32)
parser.add_argument('--latent_sizeS', type=int, default=32)
parser.add_argument('--latent_sizeC', type=int, default=32)
parser.add_argument('--depthGS', type=int, default=8)
parser.add_argument('--depthGC', type=int, default=5)
parser.add_argument('--depthT', type=int, default=3)


parser.add_argument('--reduce_dim', action="store_true")
parser.add_argument('--use_atomic', action="store_false")
parser.add_argument('--sum_pool', action="store_false")
parser.add_argument('--use_class', action="store_true")
parser.add_argument('--use_edit', action="store_true")
parser.add_argument('--use_node_embed', action="store_true")
parser.add_argument('--use_brics', action="store_true")
parser.add_argument('--update_embed', action="store_true")
parser.add_argument('--use_product', action='store_false')
parser.add_argument('--network_type', type=str, default='gcn')
parser.add_argument('--use_latent_attachatom', action="store_true")
parser.add_argument('--use_attachatom', action='store_true')
parser.add_argument('--use_feature', action="store_false")    
parser.add_argument('--use_match', action="store_true")
parser.add_argument('--use_mess', action="store_true")
parser.add_argument('--use_tree', action="store_true")
parser.add_argument('--lr', type=float, default=2)
parser.add_argument('--num', type=int, default=20)
parser.add_argument('--clip_norm', type=float, default=50.0)

parser.add_argument('--epoch', type=int, default=50)
args = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
vocab = Vocab(vocab)
avocab = common_atom_vocab

args.depthG = args.depthGC
args.hidden_size = args.hidden_sizeC
args.latent_size = args.latent_sizeC
args.embed_size = args.embed_sizeC

model_center = MolCenter(vocab, avocab, args)
try:
    model_center.load_state_dict(torch.load(args.mod_cen_path, map_location=torch.device(device)))
except:
    raise ValueError("model does not exist")

args.depthG = args.depthGS
args.hidden_size = args.hidden_sizeS
args.latent_size = args.latent_sizeS
args.embed_size = args.embed_sizeS

# synthon completion module does not use brics
tmp1 = args.use_brics
tmp2 = args.use_tree
args.use_tree = False
args.use_brics = False
model_synthon = MolSynthon(vocab, avocab, args)
try:
    model_synthon.load_state_dict(torch.load(args.mod_syn_path, map_location=torch.device(device)))
except:
    raise ValueError("model does not exist")
args.use_brics = tmp1
args.use_tree = tmp2

data = []
with open(args.test_path) as f:
    for line in f.readlines()[1:]:
        s = line.strip("\r\n ").split(",")
        smiles = s[2].split(">>")
        data.append((int(s[0]), s[1], smiles[1], smiles[0]))

output = []
start = int(args.start)
end = int(args.size) + start if args.size > 0 else len(data)

output_path = args.save_dir + args.output + "_" + str(args.knum) + "_%d" %(start) + "_%d" % (end)

res_file = open("%s_res.txt" % (output_path), 'w')
error_file = open("%s_error.txt" % (output_path), 'w')
pred_file = open("%s_pred.txt" % (output_path), 'w')

loader = MolTreeFolder(data[start:end], vocab, avocab, args.batch_size, test=True, use_atomic=True, use_class=args.use_class, use_brics=args.use_brics, use_feature=True, del_center=True, usepair=False)

top_10_bool = np.zeros((len(loader.prod_list), 10))
prod_list = []
num = 0

# optimize input molecules
for batch in loader:
    classes, product_batch, product_tree, reacts_smiles, product_smiles, synthon_smiles, skip_idxs = batch
    
    with torch.no_grad():
        top_k_trees, buffer_log_probs = model_center.test(classes, product_batch, product_tree, reacts_smiles, knum=args.knum)
        top_k_synthon_batch = [None for _ in top_k_trees]
        
        test_product_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(tree.smiles)) for tree in product_tree]
        for i, trees in enumerate(top_k_trees):
            _, tensors = MolTree.tensorize(trees, vocab=vocab, istest=True, use_atomic=True, use_feature=args.use_feature, avocab=avocab, product=False)
            top_k_synthon_batch[i] = tensors
            
        pre_react_smiles, pre_react_logs = model_synthon.test_synthon_beam_search(classes, product_batch, product_tree, top_k_trees, top_k_synthon_batch, \
                                                                                  buffer_log_probs, knum=10, product_smiles=test_product_smiles)
        
    
    for i, react_smile in enumerate(reacts_smiles):   
        for j in range(len(pre_react_smiles[i])):
            pre_smile = pre_react_smiles[i][j]
            
            if is_sim(pre_smile, react_smile):
                top_10_bool[num+i, j:] = 1
                print("%s match (%.2f)" % (product_smiles[i][0], pre_react_logs[i][j]))
                break
            else:
                string = "%s: %s fail to match %s with %s (%.2f)\n" % (product_smiles[i][0], product_smiles[i][1], react_smile, pre_smile, pre_react_logs[i][j])
                print(string)
                error_file.write(string)

    for i, react_smile in enumerate(reacts_smiles):   
        for j in range(len(pre_react_smiles[i])):
            pre_smile = pre_react_smiles[i][j]
            pred_file.write("%d %s %s %s %s %.2f\n" % (i + num, product_smiles[i][0], product_smiles[i][1], react_smile, pre_smile, pre_react_logs[i][j]))
    
    batch_10_acc = np.mean(top_10_bool[num:num+len(product_smiles), :], axis=0)
    print("iter: top 10 accuracy: %.4f  %.4f  %.4f  %.4f  %.4f  %.4f" % (batch_10_acc[0], batch_10_acc[1], batch_10_acc[2], batch_10_acc[3], batch_10_acc[4], batch_10_acc[-1]))
    
    cumu_10_acc = np.mean(top_10_bool[:num+len(product_smiles), :], axis=0)
    print("iter: top 10 accuracy: %.4f  %.4f  %.4f  %.4f  %.4f  %.4f" % (cumu_10_acc[0], cumu_10_acc[1], cumu_10_acc[2], cumu_10_acc[3], cumu_10_acc[4], cumu_10_acc[-1]))
    for i, (idx, prod) in enumerate(product_smiles):
        string = "%s %s %s\n" % (idx, prod, " ".join([str(top_10_bool[num+i, j]) for j in range(10)]))
        res_file.write(string)
    num += len(product_smiles)
    sys.stdout.flush()
    error_file.flush()
    res_file.flush()
    pred_file.flush()
    
res_file.close()
error_file.close()
pred_file.close()

top_10_acc = np.sum(top_10_bool, axis=0) / len(loader.prod_list)
print("top 10 accuracy: %.4f  %.4f  %.4f  %.4f  %.4f  %.4f" % (top_10_acc[0], top_10_acc[1], top_10_acc[2], top_10_acc[3], top_10_acc[4], top_10_acc[-1]))
