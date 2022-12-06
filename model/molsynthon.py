import random
import time
import copy
import torch
import torch.nn as nn
import numpy as np
from rdkit.Chem import Descriptors
from mol_enc import MolEncoder
from mol_dec import MolDecoder

from mol_tree import MolTree, identify_react_ids
from chemutils import set_atommap, copy_edit_mol, add_chirality, get_smiles, graph_to_mol, get_mol, get_idx_from_mapnum, get_mapnum_from_idx
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from config import device, BOND_SIZE, VALENCE_NUM, REACTION_CLS, SUB_CHARGE_NUM,  HYDROGEN_NUM, IS_RING_NUM, IS_CONJU_NUM, IS_AROMATIC_NUM
from nnutils import variable_CE_loss, get_likelihood, variable_likelihood, create_pad_tensor, index_select_ND
from sklearn.metrics import recall_score

import pdb

def make_cuda(tensors, product=True):
    make_tensor = lambda x: x if type(x) is torch.Tensor else torch.tensor(x, requires_grad=False)
    
    new_tensors1 = [make_tensor(x).to(device).long() for x in tensors[:6]]
    
    if len(tensors) > 7:
        new_tensors2 = [tensors[-3], make_tensor(tensors[-2]).to(device), make_tensor(tensors[-1]).to(device)]
        tensors = new_tensors1 + new_tensors2
    elif not product:
        tensors = new_tensors1 + [tensors[-1], None, None]
    else:
        tensors = new_tensors1 + [tensors[-1]]
      
    return tensors


class MolSynthon(nn.Module):
    """ model used to optimize molecule
    """
    def __init__(self, vocab, avocab, args):
        super(MolSynthon, self).__init__()
        self.vocab = vocab
        self.avocab = avocab
        self.hidden_size = args.hidden_size
        self.latent_size = args.latent_size
        self.atom_size = atom_size = avocab.size()
        self.use_feature = args.use_feature
        self.use_tree = args.use_tree
        self.reduce_dim = args.reduce_dim
        self.use_latent_attachatom = args.use_latent_attachatom
        self.use_product = args.use_product
        self.use_class = args.use_class
        # embedding for substructures and atoms
        self.E_a = torch.eye(atom_size).to(device)
        if args.use_feature:
            self.E_fv = torch.eye( VALENCE_NUM ).to(device)
            
            self.charge_offset = int((SUB_CHARGE_NUM - 1) / 2)
            self.E_fg = torch.eye( SUB_CHARGE_NUM ).to(device)
            self.E_fh = torch.eye( HYDROGEN_NUM ).to(device)
            self.E_fr = torch.eye( IS_RING_NUM ).to(device)
            self.E_fc = torch.eye( IS_CONJU_NUM ).to(device)
            self.E_fa = torch.eye( IS_AROMATIC_NUM ).to(device)
            
        self.E_b = torch.eye(BOND_SIZE).to(device)
       
        if self.use_feature:
            feature_embedding = (self.E_a, self.E_fv, self.E_fg, self.E_fh, self.E_fr, self.E_fc, self.E_fa, self.E_b)
        else:
            feature_embedding = (self.E_a, self.E_b)
        
        # encoder
        tmp = args.use_class
        args.use_class = False
        self.encoder = MolEncoder(self.atom_size, feature_embedding, args=args)
        
        # decoder
        args.use_class = tmp
        self.decoder = MolDecoder(vocab, avocab, self.encoder, self.charge_offset, args=args)
        
    def encode(self, tensors, product=False, usemask=False):
        """ Encode the molecule during the test
        
        Args:
            tensors: input embeddings
            orders:  
            
        Returns:
        """
        if self.use_feature: tensors[0][0][:,2] = tensors[0][0][:,2] + self.charge_offset
        #print("%.2f %.2f" % (torch.min(tensors[0][0][:, 2]), torch.max(tensors[0][0][:, 2])))
        
        mol_rev_vecs, mol_atom_vecs, mol_mess_vecs = self.encoder(tensors, product=product, usemask=usemask, use_feature=self.use_feature)
        return mol_rev_vecs, mol_atom_vecs
    
    def test_synthon(self, classes, product_batch, synthon_batch, product_trees, synthon_trees, augment=False, knum=1, product_smiles=None):
        """
        Args:
            product_batch: features of molecule X
            reacts_batch: features of molecule y
            product_trees: list of trees of molecules x
            reacts_trees: list of trees of molecules y
        """
        # prepare feature embeddings
        synthon_graphs, synthon_tensors, _ = synthon_batch

        synthon_tensors = make_cuda(synthon_tensors[0], product=False)
        # encoding
        product_graphs, product_tensors, _, _ = product_batch
        product_tensors = make_cuda(product_tensors[0], product=True)
        
        product_embed_vecs, product_atom_vecs = self.encode([product_tensors], product=True, usemask=False)
        
        product_offset, synthon_offset = 1, 1
        for i, synthon_tree in enumerate(synthon_trees):
            product_tree = product_trees[i]
            
            center_idxs = []
            idx = 0
            for node in product_tree.mol_graph.nodes:
                if 'attach' in product_tree.mol_graph.nodes[node]:
                    center_idxs.append(product_tree.mol_graph.nodes[node]['idx'])
                    
                    if self.use_product and self.use_latent_attachatom:
                        product_center_idxs[i, idx] = node + product_offset
                        idx += 1
            
            product_offset += len(product_tree.mol_graph.nodes)
            synthon_tree.finished = False
            synthon_tree.stack = []
            mapnum_to_idx = get_idx_from_mapnum(synthon_tree.mol)
            
            for j, idx in enumerate(center_idxs):
                if idx in mapnum_to_idx:
                    atom_id = mapnum_to_idx[idx]
                    synthon_tree.stack.append(atom_id)
                    if self.use_latent_attachatom:
                        synthon_center_idxs[i, j] = atom_id + synthon_offset
            
            synthon_offset += len(synthon_tree.mol_graph.nodes)
        
        if not augment:
            top_k_react_data = [(classes, synthon_tensors, synthon_graphs, synthon_trees)]
            buffer_log_probs = [[0] for _ in range(len(synthon_trees))]
        else:
            new_synthon_trees = []
            for tree in synthon_trees:
                if len(tree.stack) >= 2:
                    new_tree.a
            top_k_react_data = [(classes, synthon_tensors, synthon_graphs, synthon_trees)]
            buffer_log_probs = [[0] for _ in range(len(synthon_trees))]
            
        top_k_reacts, top_k_lhs = self.decoder.decode(product_embed_vecs, top_k_react_data, buffer_log_probs, num_k = knum, product_smiles=product_smiles)
        top_k_smiles = [[] for _ in product_trees]
        for i, trees in enumerate(top_k_reacts):
            for tree in trees:
                smiles_chiral = add_chirality(product_smiles[i], tree.smiles)
                top_k_smiles[i].append(smiles_chiral)
        
        return top_k_smiles
        
        
    def test_synthon_beam_search(self, classes, product_batch, product_trees, topk_synthons_trees, topk_synthons_batch, buffer_log_probs, knum=1, product_smiles=None):
        """
        Args:
            product_batch: features of molecule X
            reacts_batch: features of molecule y
            product_trees: list of trees of molecules x
            reacts_trees: list of trees of molecules y
        """
             
        product_graphs, product_tensors, product_ggraph, _ = product_batch
        product_tensors = product_tensors[0]
        product_graphs  = product_graphs[0]
        
        if not self.use_feature:
            product_tensors = (product_tensors[0][:, 0], product_tensors[1][:, :3], ) + product_tensors[2:]
        
        
        product_tensors = make_cuda(product_tensors)
        product_tensors = product_tensors + [None, None]
        product_embed_vecs, product_atom_vecs = self.encode([product_tensors], product=False, usemask=False)
        
        top_k_react_data = []
        for i, synthon_trees in enumerate(topk_synthons_trees):
            synthon_graphs, synthon_tensors, _ = topk_synthons_batch[i]
            
            synthon_tensors = make_cuda(synthon_tensors[0], product=False)
            
            top_k_react_data.append((classes, synthon_tensors, synthon_graphs, synthon_trees))
        
        
        top_k_reacts, top_k_lhs = self.decoder.decode(product_embed_vecs, top_k_react_data, buffer_log_probs, num_k = knum, product_smiles=product_smiles)
        
        top_k_smiles = [[] for _ in range(len(top_k_reacts))]
        for i, trees in enumerate(top_k_reacts):
            for tree in trees:
                smiles_chiral = add_chirality(product_smiles[i], tree.smiles)
                top_k_smiles[i].append(smiles_chiral)
        return top_k_smiles, top_k_lhs
        
    
    def get_synthon_padatom_vecs(self, react_tree, synthon_tree, react_atom_vecs, synthon_atom_vecs):
        react_dict = get_mapnum_from_idx(react_tree.mol)
        synthon_dict = get_idx_from_mapnum(synthon_tree.mol)
        
        pad_atom_vecs = torch.zeros_like(react_atom_vecs).to(device)
        for i in range(react_atom_vecs.shape[0]):
            mapnum = react_dict[i]
            if mapnum == 0: continue
            idx = synthon_dict[mapnum]
            
            pad_atom_vecs[i, :] = synthon_atom_vecs[idx, :]
        return pad_atom_vecs


    def forward(self, classes, product_batch, react_batch, product_trees, react_trees, total_step):
        """
        Args:
            x_batch: features of molecule X
            y_batch: features of molecule y
            x_trees: list of trees of molecules x
            y_trees: list of trees of molecules y
            beta   : weight for kl loss
        """
        react_graphs, cpu_react_tensors, react_orders = react_batch
        react_tensors = make_cuda(cpu_react_tensors)
        
        product_graphs, product_tensors, _, _ = product_batch
        
        product_tensors = make_cuda(product_tensors[0])
        product_embed_vecs, product_atom_vecs = self.encode([product_tensors], product=True, usemask=False)
        
        # decoding
        react_data = (classes, react_tensors, react_orders, react_graphs, react_trees)
        frag_loss, frag_acc, frag_rec, frag_num = self.decoder( product_embed_vecs, react_data, total_step )
        # loss
        total_loss = torch.sum(torch.stack(frag_loss, dim=0))
        
        return total_loss, frag_loss, frag_acc, frag_rec, frag_num 
