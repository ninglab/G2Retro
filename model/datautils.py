import pdb
import numpy as np
import pandas as pd
import torch
import os, random, re
import pickle
from config import device
from functools import partial
from multiprocessing import Pool
from vocab import Vocab
from torch.utils.data import Dataset, DataLoader
from mol_tree import MolTree, update_revise_atoms, identify_revise_edges
from chemutils import get_mol, get_smiles, set_atommap, get_synthon_from_smiles, canonicalize

class PairTreeFolder(object):

    def __init__(self, path, vocab, avocab, args, is_train_center=False):
        self.vocab = vocab
        self.avocab = avocab
        self.is_train_center = is_train_center
        
        self.batch_size = args.batch_size
        self.num_workers = args.ncpu
        self.shuffle = args.shuffle
        self.total_step = args.total_step
        self.total_epoch = args.epoch
        self.use_feature = args.use_feature
        self.use_brics = args.use_brics
        self.use_class = args.use_class
        self.use_atomic= args.use_atomic
        self.path = path
        self.epoch = 0
        self.step = 0
        
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                self.mol_trees = pickle.load(f)
            self.files = [None]
        else:
            self.files = [f for f in os.listdir(path) if "tensor" in f and "pkl" in f]
            
    def __iter__(self):
        if not self.shuffle:
            batches_data = []
        
        unfinish = True
        while unfinish:
            for path in self.files:
                if path is None: mol_trees = self.mol_trees
                else:
                    mol_trees = None
                    with open(self.path + path, 'rb') as f:
                        mol_trees = pickle.load(f)
                        
                        mol_trees = [(mol_tree, None) for mol_tree in mol_trees]
                        
                if self.shuffle or (not self.shuffle and i == 0):
                    batches = [mol_trees[j : j + self.batch_size] for j in range(0, len(mol_trees), self.batch_size)]
                    
                    if len(batches[-1]) < self.batch_size:
                        batches.pop()
                    
                    dataset = PairTreeDataset(batches, self.vocab, self.avocab, use_atomic=self.use_atomic, use_class=self.use_class,\
                                          use_brics=self.use_brics, use_feature=self.use_feature, is_train_center=self.is_train_center)
                    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x:x[0])

                    
                    for b in dataloader:
                        if not self.shuffle:
                            batches_data.append(b)
                       
                         
                        yield (b, self.epoch)
                        
                        self.step += 1
                        if self.step > self.total_step and self.total_step > 0:
                            unfinish = False
                            
                            break
                    
                    del batches, dataset, dataloader
                    
                    if self.shuffle and path is None:
                        random.shuffle(self.mol_trees)
                else:
                    for b in batches_data:
                        self.step += 1
                        yield (b, self.epoch)
                        if self.step > self.total_step:
                            unfinish = False
                            break
            
            if self.epoch > self.total_epoch and self.total_epoch > 0: unfinish = False
            self.epoch += 1
            if not unfinish: break
            
class MolTreeFolder(object):  
    def __init__(self, data, vocab, avocab, num_workers=10, batch_size=32, use_atomic=False, use_class=False, test=False, del_center=True, use_brics=False, usepair=False, use_feature=True, shuffle=False):
        self.batch_size = batch_size
        self.vocab = vocab
        self.avocab = avocab
        self.shuffle = shuffle
        self.usepair = usepair
        self.use_brics = use_brics
        self.use_feature = use_feature
        self.test = test
        self.type_list = [data[i][0] for i in range(len(data))]
        self.idx_list = [data[i][1] for i in range(len(data))]
        self.del_center = del_center
        self.use_class = use_class
        self.use_atomic = use_atomic
        self.num_workers = num_workers
        if self.test:
            self.reacts_list = [data[i][3] for i in range(len(data))]
            self.prod_list = [data[i][2] for i in range(len(data))]
        else:
            self.reacts_list = []
            for i in range(len(data)):
                reacts_smiles = data[i][3]
                self.reacts_list.append(reacts_smiles)
            self.prod_list = [data[i][2] for i in range(len(data))]
          
    def __iter__(self):
        batches = []
        
        for i in range(0, len(self.prod_list), self.batch_size):
            if self.use_class:
                batch = [(self.type_list[j], self.idx_list[j], self.prod_list[j], self.reacts_list[j]) for j in range(i, min(i + self.batch_size, len(self.prod_list)))]
            else:
                batch = [(None, self.idx_list[j], self.prod_list[j], self.reacts_list[j]) for j in range(i, min(i + self.batch_size, len(self.prod_list)))]
            
            batches.append(batch)
        
        dataset = MolTreeDataset(batches, self.vocab, self.avocab, use_atomic=self.use_atomic, use_class=self.use_class, test=self.test, del_center=self.del_center, use_brics=self.use_brics, usepair=self.usepair, use_feature=self.use_feature)
        
        dataset.__getitem__(1)
        
        dataloader = DataLoader(dataset, batch_size=1, num_workers=self.num_workers, shuffle=False, collate_fn=lambda x:x[0])
                
        for b in dataloader:
            yield b
        
        del batches, dataset, dataloader


class PairTreeDataset(Dataset):

    def __init__(self, data, vocab, avocab, use_class=False, use_atomic=False, is_train_center=False, use_brics=False, use_feature=False):
        self.data = data
        self.vocab = vocab
        self.avocab = avocab
        self.use_feature = use_feature
        self.use_brics = use_brics
        self.use_class = use_class
        self.use_atomic = use_atomic
        self.is_train_center = is_train_center
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        batch_data = [tmp[0] for tmp in self.data[idx]]
        
        if self.is_train_center:

            if self.use_class:
                classes = [tmp[1] for i, tmp in enumerate(self.data[idx])]
            else:
                classes = None
             
            product_tree_batch = [dpair[0] for dpair in batch_data]
            idxs = [i for i, tree in enumerate(product_tree_batch) if len(tree.order) == 0 or '2H' in tree.smiles]
            if self.use_class: classes = [cls for i, cls in enumerate(classes) if i not in idxs]
            product_tree_batch, product_batch = MolTree.tensorize(product_tree_batch, self.vocab, self.avocab, skip_mols=idxs, use_atomic=self.use_atomic, use_feature=self.use_feature, use_brics=self.use_brics, product=True)
            
            return classes, product_batch, product_tree_batch
        else:
            product_tree_batch = [dpair[0] for dpair in batch_data]
            synthon_tree_batch = [dpair[1] for dpair in batch_data]
            reacts_tree_batch  = [dpair[2] for dpair in batch_data]
            
            idxs = [i for i, tree in enumerate(product_tree_batch) if len(tree.order) == 0 or '2H' in reacts_tree_batch[i].smiles]
            
            for i, tree in enumerate(reacts_tree_batch):
                for _, idx in tree.order:
                    try:
                        if idx >= 0:
                            label = self.vocab[tree.mol_tree.nodes[idx]['label']]
                    except:
                        idxs.append(i)
                        break
            
            
            product_tree_batch, product_batch = MolTree.tensorize(product_tree_batch, self.vocab, self.avocab, skip_mols=idxs, use_atomic=self.use_atomic, istest=True, use_feature=self.use_feature, use_brics=False, product=True)
            
            reacts_tree_batch, reacts_batch = MolTree.tensorize(reacts_tree_batch, self.vocab, self.avocab, skip_mols=idxs, use_atomic=self.use_atomic, use_feature=self.use_feature, product=False)
            
            if self.use_class:
                classes = [tmp[1] for i, tmp in enumerate(self.data[idx]) if i not in idxs]
            else:
                classes = None
            
            return classes, product_batch, reacts_batch, product_tree_batch, reacts_tree_batch


class MolTreeDataset(Dataset):

    def __init__(self, data, vocab, avocab, use_class=False, del_center=True, use_atomic=False, test=False, use_brics=False, usepair=False, use_feature=False):
        self.data = data
        self.vocab = vocab
        self.avocab = avocab
        self.usepair = usepair
        self.test = test
        self.use_class = use_class
        self.use_brics = use_brics
        self.use_feature = use_feature
        self.del_center = del_center
        self.use_atomic = use_atomic
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        all_smiles = self.data[idx]
        product_trees, synthons_trees, react_smiles, target_idxs = [], [], [], []
        reaction_clses, product_smiles, synthon_smiles, skip_idxs = [], [], [], []
        
        for i, (cls, idx, prod_smile, react_smile) in enumerate(all_smiles):
            product_smiles.append( (idx, prod_smile) )
            react_smiles.append(react_smile)

            react_mol = get_mol(react_smile)
            mol, synthon_smile = get_synthon_from_smiles(react_smile)
            synthon_smiles.append(synthon_smile)
            
            if '2H' in react_smiles:
                skip_idxs.append(i)
                continue
            
            try:
                tree = MolTree(prod_smile, use_brics=self.use_brics)
            except Exception as e:
                skip_idxs.append(i)
                continue
            
            if self.usepair or not self.del_center:
                react_tree = MolTree(react_smile)
                synthon_tree = MolTree(synthon_smile)
            
                try:
                    update_revise_atoms(tree, react_tree)
                except Exception as e:
                    skip_idxs.append(i)
                    continue
                 
                synthons_trees.append(synthon_tree)
            
            product_trees.append(tree)
            reaction_clses.append(cls)
        
        
        _, product_batch = MolTree.tensorize(product_trees, self.vocab, self.avocab, istest=self.test, use_atomic=self.use_atomic, use_brics=self.use_brics, product=True, use_feature=self.use_feature)
        
        if self.del_center:
            for tree in product_trees:
                for node in tree.mol_graph.nodes:
                    if 'attach' in tree.mol_graph.nodes[node]:
                        del tree.mol_graph.nodes[node]['attach']
                 
                for idx1, idx2 in tree.mol_graph.edges:
                    if 'delete' in tree.mol_graph[idx1][idx2]:
                        del tree.mol_graph[idx1][idx2]['delete']
                    if 'change' in tree.mol_graph[idx1][idx2]:
                        del tree.mol_graph[idx1][idx2]['change']
        
        if self.use_class:
            select_clses = reaction_clses
        else:
            select_clses = None
        
        if self.usepair:
            synthons_trees, synthon_batch = MolTree.tensorize(synthons_trees, self.vocab, self.avocab, use_atomic=self.use_atomic, use_brics=False, product=False, istest=True, usemask=False, use_feature=self.use_feature)
            return select_clses, product_batch, synthon_batch, product_trees, synthons_trees, react_smiles, product_smiles, synthon_smiles, skip_idxs
        else:
            return select_clses, product_batch, product_trees, react_smiles, product_smiles, synthon_smiles, skip_idxs
