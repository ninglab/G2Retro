#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pdb
import time
import torch
import torch.nn as nn
from mol_tree import MolTree
from chemutils import BOND_LIST
from nnutils import index_select_ND, GCN, GRU, create_pad_tensor
from config import REACTION_CLS

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MolEncoder(nn.Module):
    def __init__(self, atom_size, feature_embedding, args=None):
        super(MolEncoder, self).__init__()
        self.hidden_size = args.hidden_size
        self.latent_size = args.latent_size
        self.atom_size = atom_size
        self.bond_size = len(BOND_LIST)
        self.depthT = args.depthT
        self.depthG = args.depthG
        self.use_feature = args.use_feature
        self.use_tree = args.use_tree
        self.network_type = args.network_type
        self.use_node_embed = args.use_node_embed
        self.use_class = args.use_class
        self.E_a = feature_embedding[0]
        self.E_b = feature_embedding[-1-int(self.use_class)]
        self.use_atomic = args.use_atomic
        self.sum_pool = args.sum_pool
                
        if self.use_feature:
            self.E_fv = feature_embedding[1]
            self.E_fg = feature_embedding[2]
            self.E_fh = feature_embedding[3]
            self.E_fr = feature_embedding[4]
            self.E_fc = feature_embedding[5]
            self.E_fa = feature_embedding[6]
            bond_size = sum([feature_embedding[i].shape[1] for i in range(4, 8)])
            feature_size = sum([feature_embedding[i].shape[1] for i in range(len(feature_embedding)-3-int(self.use_class))]) + 2 + int(self.use_atomic)
        else:
            bond_size = self.E_b.shape[1]
            if self.use_tree: feature_size = sum([feature_embedding[i].shape[1] for i in range(1+int(self.use_class))])
            else: feature_size = sum([feature_embedding[i].shape[1] for i in range(1)])
        
        if self.use_class:
            self.reactions = feature_embedding[-1].to(device)
            feature_size += REACTION_CLS
        
        # Parameters for Atom-level Message Passing
        input_size = feature_size + bond_size
        if self.network_type == "gcn":
            self.ampn = GCN(input_size, self.hidden_size, self.depthG)
        elif self.network_type == 'gru':
            self.ampn = GRU(input_size, self.hidden_size, self.depthG)
        
        self.outputAtom = nn.Sequential(
            nn.Linear(feature_size + self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)).to(device)
        
        # Parameters for Tree-level Message Passing
        if self.use_tree:
            self.W_i = nn.Sequential(
                nn.Linear( 2 * self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size)).to(device)
            
            self.W_g = nn.Sequential(
                nn.Linear(2 * self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size)).to(device)
            
            
            if self.network_type == 'gcn':
                self.nmpn = GCN( self.hidden_size, self.hidden_size, self.depthT )
            else:
                self.nmpn = GRU( self.hidden_size, self.hidden_size, self.depthT )
            
            self.outputNode = nn.Sequential(
                nn.Linear( 2 * self.hidden_size, self.hidden_size).to(device),
                nn.ReLU(), 
                nn.Linear(self.hidden_size, self.hidden_size)).to(device)
    
    def embed_tree(self, tree_tensors, hatom, product=False, usemask=False):
        """ Prepare the embeddings for tree message passing.
        Incoprate the learned embeddings for atoms into the tree node embeddings
        
        Args:
            tree_tensors: The data of junction tree
            hatom: The learned atom embeddings through graph message passing 
        
        """
        if len(tree_tensors) == 7:
            fnode, fmess, agraph, bgraph, cgraph, dgraph, _ = tree_tensors
        else:
            fnode, fmess, agraph, bgraph, cgraph, dgraph, _, node_mask, edge_mask = tree_tensors
            if usemask:
                agraph = (agraph * index_select_ND(edge_mask, 0, agraph).squeeze(-1)).long()
                bgraph = (bgraph * index_select_ND(edge_mask, 0, bgraph).squeeze(-1)).long()
       
        # combine atom embeddings with node embeddings 
        
        hnode = index_select_ND(hatom, 0, dgraph).sum(dim=1)
        
        if not product and usemask: hnode = hnode * node_mask
        
        # combine atom embeddings with edge embeddings
        hmess1 = hnode.index_select(index=fmess[:,0], dim=0)
        hmess2 = index_select_ND(hatom, 0, cgraph).sum(dim=1)
        
        hmess = self.W_g( torch.cat([hmess1, hmess2], dim=-1) )
        
        if len(tree_tensors) > 7 and usemask: hmess = hmess * edge_mask
        
        return hnode, hmess, agraph, bgraph
    
    def embed_atom_feature(self, fnode, classes=None, charge_set=0, use_feature=False, scopes=None):
        if use_feature and self.use_feature:
            hnode1 = self.E_a.index_select(index=fnode[:, 0], dim=0)
            hnode2 = self.E_fv.index_select(index=fnode[:, 1], dim=0)
            hnode3 = self.E_fg.index_select(index=fnode[:, 2]+charge_set, dim=0)
            hnode4 = self.E_fh.index_select(index=fnode[:, 3], dim=0)
            hnode5 = self.E_fr.index_select(index=fnode[:, 4], dim=0)
            hnode6 = self.E_fa.index_select(index=fnode[:, 5], dim=0)
            
            if self.use_atomic:
                hnode = torch.cat( (hnode1, hnode2, hnode3, hnode4, hnode5, hnode6, fnode[:, 6].unsqueeze(1)), dim=1)
            else:
                hnode = torch.cat( (hnode1, hnode2, hnode3, hnode4, hnode5, hnode6), dim=1)
        else:
            try:
                hnode = self.E_a.index_select(index=fnode, dim=0)
            except:
                pdb.set_trace()
        
        if self.use_class and classes is not None:
            cls_idxs = torch.zeros((fnode.shape[0],), dtype=torch.int32).to(device)
            for i, scope in enumerate(scopes):
                try:
                    cls_idxs[scope[0]:scope[0]+scope[1]] = classes[i]-1
                except:
                    pdb.set_trace()
            hnode7 = self.reactions.index_select(index=cls_idxs, dim=0)
            hnode = torch.cat( (hnode, hnode7), dim=1)
        
        return hnode
        
    def embed_bond_feature(self, fmess):
        if self.use_feature:
            fmess1 = self.E_b.index_select(index=fmess[:, 2], dim=0)
            fmess2 = self.E_fr.index_select(index=fmess[:, 3], dim=0)
            fmess3 = self.E_fc.index_select(index=fmess[:, 4], dim=0)
            fmess4 = self.E_fa.index_select(index=fmess[:, 5], dim=0)
            
            hmess = torch.cat( (fmess1, fmess2, fmess3, fmess4), dim=1)
        else:
            hmess = self.E_b.index_select(index=fmess[:, 2], dim=0)
        
        return hmess

    
    def embed_graph(self, graph_tensors, product=False, charge_set=0, usemask=False, use_feature=False, classes=None, subatoms=None):
        """ Prepare the embeddings for graph message passing.
        
        Args:
            graph_tensors: The data of molecular graphs
        
        """
        if len(graph_tensors) == 7:
            fnode, fmess, agraph, bgraph, _, _, scopes = graph_tensors
            hnode = self.embed_atom_feature(fnode, classes=classes, charge_set=charge_set, use_feature=use_feature, scopes=scopes)
        else:
            fnode, fmess, agraph, bgraph, _, _, scopes, atom_mask, bond_mask = graph_tensors
            hnode = self.embed_atom_feature(fnode, classes=classes, charge_set=charge_set, use_feature=use_feature, scopes=scopes)
            if usemask:
                select_hnode = index_select_ND(hnode, 0, atom_mask.nonzero()[:, 0])
                agraph = index_select_ND(agraph, 0, atom_mask.nonzero()[:, 0])
                fmess = index_select_ND(fmess, 0, bond_mask.nonzero()[:, 0])
                
                agraph = (agraph * index_select_ND(bond_mask, 0, agraph).squeeze(-1)).long()
                bgraph = (bgraph * index_select_ND(bond_mask, 0, bgraph).squeeze(-1)).long() 
        
        try:
            fmess1 = hnode.index_select(index=fmess[:, 0], dim=0)
        except:
            pdb.set_trace()
        fmess2 = self.embed_bond_feature(fmess)
        
        hmess = torch.cat([fmess1, fmess2], dim=-1)
        if usemask:
            ful_mess = torch.zeros((bond_mask.shape[0], hmess.shape[1])).to(device)
            ful_mess[ bond_mask.nonzero()[:, 0], :] = hmess
            
            return select_hnode, ful_mess, agraph, bgraph
        else:
            return hnode, hmess, agraph, bgraph


    def mpn(self, hnode, hmess, agraph, bgraph, is_tree, amask=None, bmask=None):
        """ Returns the node embeddings and message embeddings learned through message passing networks

        Args:
            hnode: initial node embeddings
            hmess: initial message embeddings
            agraph: message adjacency matrix for nodes. ( `agraph[i, j] = 1` represents that node i is connected with message j.)
            bgraph: message adjacency matrix for messages. ( `bgraph[i, j] = 1` represents that message i is connected with message j.)
            depth: depth of message passing
            W_m, W_n: functions used in message passing
            
        """
        if is_tree:
            messages = self.nmpn(hmess, bgraph, mask=bmask)
        else:
            messages = self.ampn(hmess, bgraph, mask=bmask)
        
        if agraph.shape[1] > 0:
            mess_nei = index_select_ND(messages, 0, agraph)
        else:
            mess_nei = torch.zeros((hnode.shape[0], 1, messages.shape[1])).to(device)
        
        
        node_vecs = torch.cat((hnode, mess_nei.sum(dim=1)), dim=-1)
        if is_tree:
            new_vecs = self.outputNode(node_vecs)
        else:
            new_vecs = self.outputAtom(node_vecs)
        
        if amask is not None:
            new_node_vecs = torch.zeros((amask.shape[0], new_vecs.shape[1])).to(device)
            new_node_vecs[amask.nonzero()[:, 0], :] = new_vecs
            return new_node_vecs, messages
        else:
            return new_vecs, messages
        
    def forward(self, tensors, subatoms=[], product=False, use_feature=False, classes=None, usemask=False):
        if len(tensors) == 3: graph_tensors, tree_tensors, ggraph = tensors
        else: graph_tensors = tensors[0]
        
        if product:
            tensors1 = self.embed_graph(graph_tensors, product=product, usemask=usemask, use_feature=use_feature, classes=classes)
            hatom, hmess = self.mpn(*tensors1,False)
            hatom[0,:] = hatom[0,:] * 0
            
            graph_scope = graph_tensors[-1]
            
            if self.use_tree:
                tensors2 = self.embed_tree(tree_tensors, hatom, product=product, usemask=usemask)
                hnode, _ = self.mpn(*tensors2,True)
                hnode[0,:] = hnode[0,:] * 0
                
                select_node_embeds = index_select_ND( hnode, 0, ggraph).squeeze(1)
                hnode_atom = self.W_i(torch.cat( (hatom, select_node_embeds), dim=1))
                
                if self.use_node_embed:
                    tree_scope = tree_tensors[-1]
                    embedding = torch.empty( (len(tree_scope), hnode.size(1)) ).to(device)
                    for i, scope in enumerate(tree_scope):
                        embedding[i, :] = hnode[scope[0]:scope[0]+scope[1]].sum(dim=0)
                        if not self.sum_pool: embedding[i, :] = embedding[i, :] / scope[1]
                    return embedding, hnode_atom, hmess
                
                else:
                    mol_indices = [[i for i in range(scope[0], scope[0]+scope[1])] for scope in graph_scope]
                    mol_indices = create_pad_tensor(mol_indices).to(device)
                    embedding = index_select_ND(hatom, 0, mol_indices).sum(dim=1)
                    return embedding, hnode_atom, hmess
            else:
                graph_scope = graph_tensors[-1]
        else:
            tensors = self.embed_graph(graph_tensors, product=product, use_feature=use_feature, usemask=usemask, classes=classes)
            hatom, hmess = self.mpn(*tensors, False)
            hatom[0,:] = hatom[0,:] * 0
            
            if usemask: hatom = hatom.clone() * graph_tensors[-2]
            
            graph_scope = graph_tensors[-3]

        
        mol_indices = [[i for i in range(scope[0], scope[0]+scope[1])] for scope in graph_scope]
        mol_indices = create_pad_tensor(mol_indices).to(device)
        
        embedding = index_select_ND(hatom, 0, mol_indices).sum(dim=1)
        return embedding, hatom, hmess
    
    def encode_atom(self, graph_tensors, charge_set=0, classes=None, use_feature=False, subatoms=[], usemask=True):
        """ return the atom emebeddings learned from MPN given graph tensors
        """
        
        tensors = self.embed_graph(graph_tensors, product=True, charge_set=charge_set, classes=classes, usemask=usemask, subatoms=subatoms, use_feature=use_feature)
        hatom, _ = self.mpn(*tensors, False, amask=graph_tensors[-2], bmask=graph_tensors[-1])
        hatom[0,:] = hatom[0,:] * 0
        
        if usemask: hatom = hatom * graph_tensors[-2]
        
        graph_scope = graph_tensors[-3]
        
        mol_indices = [[i for i in range(scope[0], scope[0]+scope[1])] for scope in graph_scope]
        mol_indices = create_pad_tensor(mol_indices).to(device)
        
        embedding = index_select_ND(hatom, 0, mol_indices).sum(dim=1)
        return embedding, hatom
        
    def encode_node(self, tree_tensors, hatom, node_idx):
        """ return the node embedding learned from MPN given tree tensors, learned atom embeddings 
        and the index of node to be learned.
        """
        hnode, hmess, agraph, bgraph = self.embed_tree(tree_tensors, hatom, product=True)
        hnode = index_select_ND(hnode, 0, node_idx)
        agraph = index_select_ND(agraph, 0, node_idx)
        hnode, _ = self.mpn(hnode, hmess, agraph, bgraph, self.depthT, self.W_t, self.outputNode)
        return hnode
