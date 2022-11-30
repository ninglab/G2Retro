import time
import torch
import copy
import math
import networkx as nx
import torch.nn as nn
import numpy as np
from rdkit import Chem
from sklearn.metrics import recall_score, accuracy_score
from nnutils import index_select_ND, GRU, MPL, bfs, unique_tensor, get_likelihood
from mol_tree import MolTree
from config import REACTION_CLS, device
from functools import partial
from multiprocessing import Pool
from chemutils import check_atom_valence, check_attach_atom_valence, get_mol, \
                      get_mapnum_from_idx, get_idx_from_mapnum, get_uniq_atoms, \
                      get_smiles, graph_to_mol, mol_to_graph, attach_mol_graph, bond_equal

import pdb

def make_cuda(tensor):
    make_tensor = lambda x: x if type(x) is torch.Tensor else torch.tensor(x, requires_grad=False)
    
    cuda_tensors = [make_tensor(x).to(device).long() for x in tensor[:-4]] + list(tensor[-4:])
   
    return cuda_tensors


        
class MolDecoder(nn.Module):
    def __init__(self, vocab, avocab, encoder, charge_set, args=None):
        super(MolDecoder, self).__init__()
        self.hidden_size = args.hidden_size
        self.latent_size = args.latent_size
        self.depthT = args.depthT
        self.vocab_size = vocab.size()
        self.vocab = vocab
        self.avocab = avocab
        self.charge_set = charge_set

        self.use_class = args.use_class 
        if args.use_class:
            self.reactions = nn.Embedding( REACTION_CLS, self.latent_size, padding_idx=0 ).to(device)
            
        self.atom_size = avocab.size()
        self.encoder = encoder
        self.reduce_dim = args.reduce_dim
        self.uniq_atom_dict = {}
        self.use_atomic = args.use_atomic      
        self.use_feature = args.use_feature
        self.update_embed = args.update_embed
        self.use_attachatom = args.use_attachatom
        self.use_product = args.use_product
        self.uniq_atom_dict = {}
        self.ncpu = args.ncpu
        
        
        if self.reduce_dim:
            self.reduce_map = nn.Sequential( nn.Linear(self.hidden_size * 2, self.hidden_size), 
                                             nn.ReLU(),
                                             nn.Linear(self.hidden_size, self.latent_size) ).to(device)
        # Parameters for function used to predict child node connections
        feature_count = 2 # atom representation; synthon representation; synthon / product representation
        
        if self.update_embed:
            feature_count += 1 # add updated synthon representation
        
        if self.use_attachatom:
            feature_count += 1 # add attach atom representation
        
        if self.use_product and not self.reduce_dim:
            feature_count += 1
        
        if self.reduce_dim:            
            layer_size = (feature_count-1) * self.hidden_size + self.latent_size
        else:
            layer_size = feature_count * self.hidden_size
        
        if args.use_class:
            layer_size += self.latent_size
        
         
        # Parameters for function used to predict connection
        self.W_t = nn.Sequential( nn.Linear(layer_size, 2 * self.hidden_size), 
                                  nn.ReLU(),
                                  nn.Linear(2 * self.hidden_size, self.hidden_size), 
                                  nn.ReLU(),
                                  nn.Linear(self.hidden_size, 1) ).to(device)
        
        # Parameters for function used to predict child node types
        self.W_n = nn.Sequential( nn.Linear(layer_size, 2 * self.hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(2 * self.hidden_size, self.hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(self.hidden_size, self.vocab_size) ).to(device)
        
        #Loss Functions
        self.node_loss = nn.CrossEntropyLoss(size_average=False)
        self.topo_loss = nn.BCEWithLogitsLoss(size_average=False)
        
    def get_target_predictions(self, graphs):
        """ Get the ground truth disconnection site labels for prediction
        """
        labels = []
        for graph in graphs:
            for idx in graph.nodes:
                if graph.nodes[idx]['target']:
                    labels.append(idx)
        return labels
    
    def apply_tree_mask(self, tensors):
        """ Mask the fragments to be added in the tree of molecule y
        so that the model can learn the fragments to be added with teacher forcing.

        Args:
            tensors: embeddings to be masked
            masks: node masks and edge masks
        """
        fnode, fmess, agraph, bgraph, cgraph, dgraph, scope, nmask, emask = tensors
        
        # substructure matrix mask
        agraph = (agraph * index_select_ND(emask, 0, agraph).squeeze(-1)).long()
        # edge matrix mask
        bgraph = (bgraph * index_select_ND(emask, 0, bgraph).squeeze(-1)).long()
        
        tensors = (fnode, fmess, agraph, bgraph, cgraph, dgraph, scope)
        return tensors, (nmask, emask)

    def apply_graph_mask(self, tensors, graphs=None, global_graph=None, trees=None):
        """ Mask the fragments to be added in the graph of molecules y
        ...
        """
        fnode, fmess, agraph, bgraph, cgraph, dgraph, scope, amask, bmask = tensors
        
        
        # atom matrix mask
        agraph = (agraph * index_select_ND(bmask, 0, agraph).squeeze(-1)).long()
        # bond matrix mask
        bgraph = (bgraph * index_select_ND(bmask, 0, bgraph).squeeze(-1)).long()
        
        mols = None
        if self.use_feature:
            fnode, fmess, mols = self.update_feature(graphs, global_graph, scope, fnode, fmess, trees=trees)
        
        tensors = (fnode, fmess, agraph, bgraph, cgraph, dgraph, scope, amask, bmask)
        return tensors, (amask, bmask), mols
    
    def update_graph_mask(self, tensors, masks, node_idx, tree_idxs, graphs=None, trees=None, global_graph=None, pointers=None):
        """ Update the graph mask after the prediction of node with idx `node_idx`,
        so that messages can be passed through the added new atoms in the new node.
        """
        amask, bmask = masks
        fnode, fmess, agraph, bgraph, egraph, dgraph, scope = tensors
        new_amask = copy.deepcopy(amask)
        
        if node_idx is not None:
            cls = index_select_ND(dgraph, 0, node_idx)
            map_tree_atoms = torch.LongTensor([i for idx in tree_idxs for i in range(scope[idx][0], scope[idx][0] + scope[idx][1])]).to(device)
        
            # Get all the atoms within the node `node_idx`
            new_amask.scatter_(0, cls[cls>0].unsqueeze(1), 1)
            
            new_amask.scatter_(0, map_tree_atoms.unsqueeze(1), 0)
        
        for i, pointer in enumerate(pointers):
            if pointer == -1 and new_amask[scope[i][0]] != 0:
                new_amask[scope[i][0]:scope[i][0]+scope[i][1]] = 0
        
        # get the new edge mask from the atom mask
        new_emask = new_amask * new_amask.transpose(0, 1)
        
        new_emask_idx = (new_emask * egraph).nonzero()
        new_emask_idx = egraph[new_emask_idx[:,0], new_emask_idx[:,1]].unsqueeze(1)
        
        new_bmask = torch.zeros(bgraph.size(0), 1, requires_grad=False).to(device)
        if torch.max(new_emask_idx) > new_bmask.shape[0]: pdb.set_trace()
        new_bmask.scatter_(0, new_emask_idx, 1)
            
        agraph = (agraph * index_select_ND(new_bmask, 0, agraph).squeeze(-1)).long()
        bgraph = (bgraph * index_select_ND(new_bmask, 0, bgraph).squeeze(-1)).long()
        
        if self.use_feature and node_idx is not None:
            atom_idxs = torch.unique(torch.flatten(cls))
            fnode, fmess, _ = self.update_feature(graphs, global_graph, scope, fnode, fmess, trees=trees, atom_idxs=atom_idxs)
            graph_tensors = [fnode, fmess, agraph, bgraph, egraph, dgraph, scope, new_amask, new_bmask]
        else:
            graph_tensors = tensors[:2]+[agraph, bgraph, egraph, dgraph, scope, new_amask, new_bmask]
        
        masks = (new_amask, new_bmask)
        return graph_tensors, masks
            
    def update_tree_mask(self, tree_batchG, tree_tensors, masks, node_idx):
        """ Update the tree mask after the prediction of node with idx `node_idx`,
        so that messages can be passed through the added new node.
        """
        nmask, emask = masks
        _, _, agraph, bgraph, cgraph, dgraph, _ = tree_tensors
        
        nmask.scatter_(0, node_idx.unsqueeze(1), 1)
        
        # Get the indices of messages/edges connected with node
        mess_idxs = []
        pairs = []
        for i in range(node_idx.size(0)):
            idx = node_idx[i].item()
            nei_idxs = [edge[1] for edge in tree_batchG.edges(idx) if nmask[edge[1]] == 1]
            pairs.extend([(idx, nei_idx) for nei_idx in nei_idxs])
        
        for pair in pairs:
            mess_idxs.append(tree_batchG[pair[0]][pair[1]]['mess_idx'])
            mess_idxs.append(tree_batchG[pair[1]][pair[0]]['mess_idx'])
        
        mess_idxs = torch.LongTensor(mess_idxs).to(device).unsqueeze(1)
        emask.scatter_(0, mess_idxs, 1)
        
        agraph = (agraph * index_select_ND(emask, 0, agraph).squeeze(-1)).long()
        bgraph = (bgraph * index_select_ND(emask, 0, bgraph).squeeze(-1)).long()
        
        tree_tensors = tree_tensors[:2]+[agraph, bgraph, cgraph, dgraph, tree_tensors[-1]]
        masks = (nmask, emask)
        return tree_tensors, masks
    
    def get_node_embedding(self, tree_tensors, node_idx, masks, hatom):
        """ Get the embeddings of nodes using the message passing networks in encoder
        Args:
            tree_tensors: the tree embebdings used for TMPN
            node_idx: the index of nodes with embeddings to be learned 
        
        """
        nmask, emask = masks
        
        fnode, fmess, agraph, bgraph, cgraph, dgraph, scope = tree_tensors
        nei_mess = index_select_ND(agraph, 0, node_idx)
        nei_mess = nei_mess[nei_mess > 0]
        new_emask = torch.zeros(emask.shape, requires_grad=False).to(device)
        for depth in range(self.depthT):
            new_nei_mess = index_select_ND(bgraph, 0, nei_mess)
            new_nei_mess = new_nei_mess[new_nei_mess > 0]
            nei_mess = torch.unique(torch.cat([nei_mess, new_nei_mess], dim=0))
        
        new_emask.scatter_(0, nei_mess.unsqueeze(1), 1)
        new_emask = torch.mul(new_emask, emask)
        
        agraph = (agraph * index_select_ND(new_emask, 0, agraph).squeeze(-1)).long()
        bgraph = (bgraph * index_select_ND(new_emask, 0, bgraph).squeeze(-1)).long()
        
        fmess = (fmess * index_select_ND(new_emask, 0, fmess).squeeze(-1)).long()
        tensors = (fnode, fmess, agraph, bgraph, cgraph, dgraph, None)
        
        # tree message passing
        hnode = self.encoder.encode_node(tensors, hatom, node_idx)
        
        return hnode
    
    def get_atom_embedding(self, tree_batch, graph_tensors, node_idx):
        """ Get the embeddings of atoms using the MPN in encoder
        """
        
        fatom, fbond, agraph, bgraph, egraph, scope, _, _ = graph_tensors
        amask = torch.zeros(fatom.size(0), 1, requires_grad=False).to(device)
        
        clusters = []
        for i in range(node_idx.size(0)):
            node = node_idx[i].item()
            cluster = tree_batch.nodes[node]['clq']
            if len(cluster) > 2:
                clusters.extend(cluster)
        
        clusters = torch.LongTensor(clusters).unsqueeze(1).to(device)
        amask.scatter_(0, clusters, 1)
        
        emask = amask * amask.transpose(0, 1)
        emask_idx = (emask * egraph).nonzero()
        emask_idx = egraph[emask_idx[:,0], emask_idx[:,1]].unsqueeze(1)
        
        emask = torch.zeros(fbond.size(0), 1, requires_grad=False).to(device)
        emask.scatter_(0, emask_idx, 1)
        agraph = agraph * index_select_ND(emask, 0, agraph).squeeze(-1).long()
        bgraph = bgraph * index_select_ND(emask, 0, bgraph).squeeze(-1).long()
        
        tensors = (fatom, fbond, agraph, bgraph, None, None)
        
        _, hatom = self.encoder.encode_atom(tensors, usemask=False)
        
        return hatom


    def update_feature(self, graphs, global_graph, scopes, atom_tensor, bond_tensor, trees=[], atom_idxs=[]):
        """ update the atom features and bond features during the generation process
        """
        all_atom_idxs = {i:[] for i in range(len(graphs))}
        
        if len(atom_idxs) > 0:
            for atom_idx in atom_idxs:
                atom_idx = atom_idx.item()
                
                for i, scope in enumerate(scopes):
                    if atom_idx >= scope[0] and atom_idx < scope[0] + scope[1]:
                        all_atom_idxs[i].append(atom_idx - scope[0])
                        break
        
        mols = []
        for i, graph in enumerate(graphs):
            if graph is None: continue
            sgg_atom_idxs = all_atom_idxs[i]
            
            if len(sgg_atom_idxs) == 0 and len(atom_idxs) > 0:
                continue
            elif len(atom_idxs) > 0 and len(sgg_atom_idxs) > 0:
                for atom1_idx, atom2_idx in graph.edges(sgg_atom_idxs):
                    if 'revise' in graph[atom1_idx][atom2_idx] and graph[atom1_idx][atom2_idx]['revise'] == 1:
                        if atom1_idx + scopes[i][0] in atom_idxs and atom2_idx + scopes[i][0] in atom_idxs:
                            graph[atom1_idx][atom2_idx]['revise'] = -1
                            graph[atom2_idx][atom1_idx]['revise'] = -1
            elif len(atom_idxs) == 0:
                for aid in graph.nodes:
                    if 'attach' in graph.nodes[aid] and graph.nodes[aid]['attach'] == 1:
                        sgg_atom_idxs.append(aid)
            
            mol, map_dict = graph_to_mol(graph, keep_atom=False, return_map=True)
            mols.append(mol)
            
            for atom_idx in sgg_atom_idxs:
                atom_gidx = atom_idx + scopes[i][0]
                    
                valence = atom_tensor[atom_gidx, 1].item()
                charge  = atom_tensor[atom_gidx, 2].item() 
                hydrogen = atom_tensor[atom_gidx, 3].item()
                is_inring = atom_tensor[atom_gidx, 4].item()
                is_aromatic = atom_tensor[atom_gidx, 5].item()
                 
                atom = mol.GetAtomWithIdx(map_dict[atom_idx])
                cur_valence = atom.GetTotalValence()
                cur_charge  = atom.GetFormalCharge()
                cur_hydrogen = atom.GetTotalNumHs()
                cur_inring = atom.IsInRing()
                cur_aromatic = atom.GetIsAromatic()
                
                atom_tensor[atom_gidx, 1:6] = torch.LongTensor([cur_valence, cur_charge, cur_hydrogen, cur_inring, cur_aromatic]).to(device)
                
                for atom1_idx, atom2_idx in graph.edges(atom_idx):
                    atom1_gidx = atom1_idx + scopes[i][0]
                    atom2_gidx = atom2_idx + scopes[i][0]
                    
                    if 'revise' in graph[atom1_idx][atom2_idx] and graph[atom1_idx][atom2_idx]['revise'] == 1 and \
                        (atom1_gidx not in atom_idxs or atom2_gidx not in atom_idxs):
                        continue
                    bond = mol.GetBondBetweenAtoms(map_dict[atom1_idx], map_dict[atom2_idx])
                    
                    cur_is_conju = bond.GetIsConjugated()
                    
                    mess_idx1 = global_graph[atom1_gidx][atom2_gidx]['mess_idx']
                    mess_idx2 = global_graph[atom2_gidx][atom1_gidx]['mess_idx']
                    
                    is_conju = bond_tensor[mess_idx1, 3]
                    
                    if cur_is_conju != is_conju:
                        bond_tensor[mess_idx1, 3] = cur_is_conju
                        bond_tensor[mess_idx2, 3] = cur_is_conju
                        
        return atom_tensor, bond_tensor, mols
        
    def forward(self, product_embed_vecs, react_data, total_step, tmp=None):
        
        batch_size = product_embed_vecs.shape[0]
        synthon_vecs = None
        
        topo_hiddens, topo_targets = [],[]
        node_hiddens, node_targets = [],[]
        
        tree_idxs = []
            
        classes, tensors, orders, graphs, trees = react_data
        
        if self.use_class:
            class_vecs = self.reactions(torch.LongTensor(classes).to(device)-1)
        
        single_graphs = [copy.deepcopy(tree.mol_graph) if tree is not None else None for tree in trees]
        
        # Get the masked tree tensors and graph tensors
        cur_graph_tensors, graph_mask, mols = self.apply_graph_mask(tensors, graphs=single_graphs, global_graph=graphs, trees=trees)
        
        attach_atom_idxs = torch.LongTensor([orders[0][0] if len(orders) > 0 else -1 for i in range(batch_size)]).to(device)
        
        maxt = max( [len(order) for order in orders] )
        off_set1, off_set2 = 1, 1
        step = 0
        pointers = [0 for _ in range(batch_size)]
        while max(pointers) >= 0:
            batch_list = [i for i in range(batch_size) if pointers[i] >= 0]
            tree_idxs = [i for i in range(batch_size) if pointers[i] < 0]
            
            pred_list, nodex_idxs = [],[]
            stop_batch_idxs, topo_labels, stop_atomx_idxs, stop_list, nodey_idxs, node_target = [],[],[],[],[],[]
            
            # Get the ground truth labels for child connection / child node type predictions.
            for i in batch_list:
                start_id = pointers[i]
                
                
                for j in range(start_id, len(orders[i])):
                    xatomid, yid, ylabel = orders[i][j]
                    
                    stop_batch_idxs.append(i)
                    stop_atomx_idxs.append(xatomid)

                    if yid >= 0:
                        pred_list.append(len(topo_labels))
                        nodey_idxs.append(yid)
                        node_target.append(ylabel)
                        topo_labels.append(0)
                        pointers[i] += 1
                        break
                    else:
                        topo_labels.append(1)
                        pointers[i] += 1

                if pointers[i] == len(orders[i]): pointers[i] = -1
            
            nodey_idxs = torch.LongTensor(nodey_idxs).to(device)
            stop_atomx_idxs = torch.LongTensor(stop_atomx_idxs).to(device)
            stop_batch_idxs = torch.LongTensor(stop_batch_idxs).to(device)

            node_target = torch.LongTensor(node_target).to(device)
            topo_labels = torch.FloatTensor(topo_labels).to(device)
            pred_list = torch.LongTensor(pred_list).to(device)
            
            # Get the Atom Embeddings learned from MPN in encoder and update graphs
            embedding, hatom1 = self.encoder.encode_atom(cur_graph_tensors, charge_set=self.charge_set, usemask=True, use_feature=self.use_feature)
            if step == 0:
                synthon_vecs = torch.cat( (embedding, product_embed_vecs), dim=1)
                if self.reduce_dim:
                    synthon_vecs = self.reduce_map(synthon_vecs)
            
            tmp_h_vecs = index_select_ND(synthon_vecs, 0, stop_batch_idxs)
            if self.use_class: tmp_r_vecs = index_select_ND(class_vecs, 0, stop_batch_idxs)

            topo_hidden1 = []
            pre_hatom1 = index_select_ND(hatom1, 0, stop_atomx_idxs)
            topo_hidden1.append(pre_hatom1)
            
            if self.update_embed:
                pre_embed = index_select_ND(embedding, 0, stop_batch_idxs)
                topo_hidden1.append(pre_embed)
            
            if self.use_attachatom:
                tmp_attach_atom_idxs = index_select_ND(attach_atom_idxs, 0, stop_batch_idxs)
                attach_atom_embed = index_select_ND(hatom1, 0, tmp_attach_atom_idxs)
                topo_hidden1.append(attach_atom_embed)
            
            topo_hidden1.extend([tmp_h_vecs])
            if self.use_class: topo_hidden1.extend([tmp_r_vecs])
            
            topo_hidden1 = torch.cat(topo_hidden1, dim=1)
            topo_hiddens.append(topo_hidden1)
            topo_targets.append(topo_labels)
            
            if nodey_idxs.shape[0] > 0:
                cur_graph_tensors, graph_mask = self.update_graph_mask(tensors[:-2], graph_mask, \
                                                                       nodey_idxs, tree_idxs, \
                                                                       graphs=single_graphs, trees=(trees, tensors[-2:]), global_graph=graphs, pointers=pointers)
            if len(pred_list) > 0:
                node_hidden = index_select_ND(topo_hidden1, 0, pred_list)
                node_targets.append(node_target)
                node_hiddens.append(node_hidden)
            step += 1
        
        node_hiddens = torch.cat(node_hiddens, dim=0)
        node_targets = torch.cat(node_targets, dim=0)
        node_scores = self.predict(node_hiddens, "node").squeeze(dim=1)
        node_loss = self.node_loss(node_scores, node_targets) / len(orders)
        _, node = torch.max(node_scores, dim=1)
        
        node_acc = torch.eq(node, node_targets).float()
        node_acc = torch.sum(node_acc) / node_targets.nelement()
        
        # child node connection Prediction
        topo_hiddens = torch.cat(topo_hiddens, dim=0)
        topo_scores = self.predict(topo_hiddens, "topo").squeeze(dim=1)
        topo_targets = torch.cat(topo_targets, dim=0)
        
        topo_loss = self.topo_loss(topo_scores, topo_targets)/len(orders)
        topo = torch.ge(topo_scores, 0).float()
        topo_acc = torch.eq(topo, topo_targets).float()
        topo_acc = torch.sum(topo_acc) / topo_targets.nelement()
        topo_rec = recall_score(topo_targets.data.to('cpu'), topo.data.to('cpu'), pos_label=0)
        
        loss = (node_loss, topo_loss)
        acc = (node_acc.item(), topo_acc.item())
        rec = (topo_rec,)
        num = (node_targets.size(0), topo_targets.size(0), topo_targets.nonzero().size(0))
        
        return loss, acc, rec, num
    
    def atom_loss(self, scores, targets1, targets2, labels):
        """ calculate the loss of predictions with scores.
        These predictions assign a score for each candidate, and predict the candidate with 
        the maximum score.

        Args:
            scores: the predicted scores for candidates of all predictions at a time step
                    for all molecules within a batch.
            targets1: the index of candidates with the maximum scores for each prediction
            targets2: the index of all candidates for each prediction
            labels: the ground truth label

        Return:
            loss: negative log likelihood loss
            acc: prediction accuracy
        """
        scores = torch.cat([torch.tensor([[0.0]]).to(device), scores], dim=0)
        scores1 = index_select_ND(scores, 0, targets1)
        scores2 = index_select_ND(scores, 0, targets2).squeeze(-1)
        
        mask = torch.zeros(scores2.size()).to(device)
        index = torch.nonzero(targets2)
        mask[index[:,0], index[:,1]] = 1
        
        loss2 = torch.sum(torch.log(torch.sum(torch.exp(scores2) * mask, dim=1)))
        loss = - torch.sum(scores1) + loss2
        
        masked_scores2 = torch.where(targets2==0, torch.FloatTensor([-10]).to(device), scores2)
        acc = torch.sum(torch.argmax(masked_scores2, dim=1) == labels).float() / labels.size(0)
        return loss, acc.item()
        
    
    def scoring(self, vector1, vector2, mode, active="tanh"):
        if self.score_func == 1:
            hidden = torch.cat((vector1, vector2), dim=1)
            scores = self.predict(hidden, mode, active=active)
        else:
            cand_vecs = self.predict(vector2, mode, active=active)
            scores = torch.bmm(vector1.unsqueeze(1), cand_vecs.unsqueeze(2)).squeeze(-1)
        return scores
    
    def predict(self, hiddens, mode, active="relu"):
        if mode == "target":
            V, U = self.W_a, self.U_a
        elif mode == "node":
            V = self.W_n
        elif mode == "topo":
            V = self.W_t
        elif mode == "delete":
            V, U = self.W_d, self.U_d
        elif mode == "atom1":
            V, U = self.W_a1, self.U_a1
        elif mode == "atom2":
            V, U = self.W_a2, self.U_a2
        else:
            raise ValueError('wrong')
        
        return V(hiddens)
   
    def insert_embed(self, embedding, atom_vecs, scopes, buffer_data):
        for i, last_data in enumerate(buffer_data):
            if last_data is not None:
                embedding[i, :] = last_data[0]
                atom_vecs[scopes[i][0]:scopes[i][0]+scopes[i][1], :] = last_data[1]
        

    def decode(self, product_vecs, top_k_react_data, buffer_log_probs, product_smiles=None, tmp=None, num_cpus=10, num_k=1):
        
        num_trees = len(buffer_log_probs)
        top_k_h_vecs = []
        top_k_trees = [[] for _ in range(num_trees)]
        top_k_log_probs = [[] for _ in range(num_trees)]
        top_k_smiles = [[] for _ in range(num_trees)]
        
        buffer_trees = [[] for _ in range(num_trees)]
        last_buffer_data  = [[] for _ in range(num_trees)]
        # generated reactants should not the same with the products
        if product_smiles is not None:
            visited_smiles = [[product_smiles[i]] for i in range(num_trees)]
        else:
            visited_smiles = [[] for _ in range(num_trees)]
        
        for i in range(len(top_k_react_data)):
            if self.reduce_dim:
                top_k_h_vecs.append( torch.zeros(num_trees, self.latent_size).to(device) )
            else:
                top_k_h_vecs.append( torch.zeros(num_trees, self.hidden_size * 2).to(device) )
                
            react_data = top_k_react_data[i]
            
            for j, tree in enumerate(react_data[-1]):
                buffer_trees[j].append(tree)
                last_buffer_data[j].append(None)
                
        buffer_zvec_idxs = [[i for i in range(len(trees))] for trees in buffer_trees]
        
        if self.use_class:
            classes = torch.LongTensor(top_k_react_data[0][0]).to(device) - 1
            cls_vecs = self.reactions(classes)
        else: cls_vecs = None
        
        finished = [0 for _ in range(num_trees)]
        tensor = top_k_react_data
        steps = 0
        while steps <= 30:
            steps += 1
            
            new_buffer_zvec_idxs = [[] for _ in buffer_trees]
            new_buffer_trees = [[] for _ in buffer_trees]
            new_buffer_log_probs = [[] for _ in buffer_trees]
            new_last_buffer_data  = [[] for _ in buffer_trees]     
            finished_trees = [[] for _ in buffer_trees]
            finished_log_probs = [[] for _ in buffer_trees]
            
            buffer_size = sum([len(tree) for tree in buffer_trees])
            if buffer_size == 0:
                break
            
            for beam_idx in range(num_k):
                
                trees = [tree[beam_idx] if len(tree) > beam_idx else None for j, tree in enumerate(buffer_trees)]
                node_atom_idxs = []
                node_hiddens = []
                 
                tree_batch_idxs = []
                tree_batches = []
                tree_zvec_idxs = []
                tree_buffer = []
                for k, tree in enumerate(trees):
                    if tree is None: continue
                    # only contain one tree
                    
                    tree_zvec_idxs.append(buffer_zvec_idxs[k][beam_idx])
                    tree_batch_idxs.append(k)
                    tree_batches.append(tree)
                    tree_buffer.append(last_buffer_data[k][beam_idx])
                    
                if len(tree_batches) == 0: break
                
                top_k_num1 = sum([len(trees) for trees in top_k_trees])
                tensor1 = MolTree.tensorize_decoding(tree_batches, self.vocab, self.avocab, tree_buffer=tree_buffer, extra_len=8, istest=True, use_atomic=self.use_atomic, use_feature=self.use_feature)
                
                tensor1 = make_cuda(tensor1)
                
                embedding, atom_vecs1 = self.encoder.encode_atom(tensor1, charge_set=self.charge_set, usemask=False, use_feature=self.use_feature)
                self.insert_embed( embedding, atom_vecs1, tensor1[-3], tree_buffer )
                if steps == 1:
                    for j in range(len(top_k_h_vecs)):
                        idxs = [k for k, z_idx in enumerate(tree_zvec_idxs) if z_idx == j]
                        batch_idxs = torch.LongTensor([tree_batch_idxs[k] for k in idxs]).to(device)
                        idxs = torch.LongTensor(idxs).to(device)
                        
                        batch_synthon_vecs = index_select_ND(embedding, 0, idxs)
                        batch_product_vecs = index_select_ND(product_vecs, 0, batch_idxs)
                        latent_h_vecs = torch.cat( (batch_synthon_vecs, batch_product_vecs), dim=1)
                        
                        if self.reduce_dim:
                            latent_h_vecs = self.reduce_map(latent_h_vecs)
                        
                        top_k_h_vecs[j][batch_idxs, :] = latent_h_vecs
                
                topo_data = (cls_vecs, tree_batches, tree_batch_idxs, tree_zvec_idxs, embedding, atom_vecs1, tensor1[-3])
                
                stop_trees, stop_log_probs, nonstop_log_probs, node_data = self.decode_topo(trees, beam_idx, top_k_trees, top_k_log_probs, \
                                                                                            top_k_h_vecs, topo_data)
                
                 
                node_hiddens, node_atomx_idxs, node_batch_idxs, node_zvec_idxs = node_data
                 
                if node_hiddens is None:
                    break
                        
                node_scores = self.predict(node_hiddens, "node").squeeze(dim=1)
                node_likelihoods = get_likelihood(node_scores).cpu().detach().numpy()
                sort_nodes = np.argsort(-node_likelihoods, axis=1)
                
                combined_items = []
                
                for k in range( len(node_atomx_idxs) ):
                    atom_idx = node_atomx_idxs[k]
                    
                    sort_node = sort_nodes[k, :]
                    node_lh = node_likelihoods[k, :]
                    
                    tree_idx = node_batch_idxs[k]
                    
                    tree = trees[tree_idx]
                    
                    combined_items.append( (steps, node_lh, sort_node, tree, atom_idx) )
                
                
                tmp_res = []
                for item in combined_items:
                    tmp1 = get_cand_trees(item, vocab=self.vocab, num_k=num_k)
                    tmp_res.append(tmp1)
                
                tmp_trees = [[] for _ in buffer_trees]
                tmp_log_probs = [[] for _ in buffer_trees]
                tmp_zvec_idxs = [-1 for _ in buffer_trees]
                tmp_atom_idxs = [-1 for _ in buffer_trees]
                
                for k, (tmp_tree, tmp_prob) in enumerate(tmp_res):
                    batch_idx = node_batch_idxs[k]
                    
                    log_prob = buffer_log_probs[batch_idx][beam_idx] + nonstop_log_probs[batch_idx]
                    
                    z_vec_idx = node_zvec_idxs[k]
                    
                    tmp_trees[batch_idx] = tmp_tree
                    
                    tmp_atom_idxs[batch_idx] = combined_items[k][4]
                    
                    tmp_log_probs[batch_idx] = [ (log_prob + prob, buffer_log_probs[batch_idx][beam_idx], nonstop_log_probs[batch_idx], prob) for prob in tmp_prob ]
                    
                    tmp_zvec_idxs[batch_idx] = z_vec_idx
                
                new_buffer_data = ( new_buffer_trees, new_buffer_log_probs, new_buffer_zvec_idxs, new_last_buffer_data )
                
                for tidx in range(len(tmp_trees)):
                    stop_tree = stop_trees[tidx]
                    if stop_tree is None: continue
                    
                    stop_log_prob = buffer_log_probs[tidx][beam_idx] + stop_log_probs[tidx]
                    log_prob = [log_probs[0] for log_probs in tmp_log_probs[tidx]]
                    
                    sort_prob_key = [k for k, prob in sorted(zip(np.arange(len(log_prob)), log_prob), reverse=True, key=lambda x: x[1]) if prob != -math.inf]
                    
                    new_zvec_idx = tmp_zvec_idxs[tidx]
                    
                    batch_tree_idx = tree_batch_idxs.index(tidx)
                    scope = tensor1[-3][batch_tree_idx]
                    self.update_buffer(stop_tree, stop_log_prob, new_zvec_idx, \
                                       new_buffer_data, finished_trees, finished_log_probs, \
                                       visited_smiles[tidx], trees, tidx, last_vecs=(embedding[batch_tree_idx,:], atom_vecs1[scope[0]:scope[0]+scope[1], :]))
                    
                    for idx in sort_prob_key[:num_k]:
                        new_tree = tmp_trees[tidx][idx]
                        new_prob = tmp_log_probs[tidx][idx][0]
                        
                        
                        self.update_buffer(new_tree, new_prob, new_zvec_idx, \
                                           new_buffer_data, finished_trees, finished_log_probs, \
                                           visited_smiles, trees, tidx)
                    
            new_buffer_trees, new_buffer_log_probs, new_buffer_zvec_idxs, new_last_buffer_data = new_buffer_data 
            for i in range(len(new_buffer_trees)):
                log_prob = new_buffer_log_probs[i] + finished_log_probs[i]
                
                num_new_buffer = len(new_buffer_log_probs[i])
                if len(log_prob) == 0:
                    buffer_trees[i] = []
                    buffer_log_probs[i] = []
                    last_buffer_data[i] = []
                    continue
                
                sort_prob_key = [k for k, prob in sorted(zip(np.arange(len(log_prob)), log_prob), reverse=True, key=lambda x: x[1]) if prob != -math.inf]
                if len(top_k_log_probs[i]) < num_k: threshold = -200
                else: threshold = sorted(top_k_log_probs[i], reverse=True)[num_k-1]
                
                buffer_zvec_idxs[i], buffer_log_probs[i], buffer_trees[i], last_buffer_data[i] = [],[],[],[]
                
                for j, idx in enumerate(sort_prob_key):
                    if idx < len(new_buffer_log_probs[i]) and new_buffer_log_probs[i][idx] > threshold:
                        buffer_zvec_idxs[i].append(new_buffer_zvec_idxs[i][idx])
                        buffer_trees[i].append(new_buffer_trees[i][idx])
                        buffer_log_probs[i].append(new_buffer_log_probs[i][idx])
                        last_buffer_data[i].append(new_last_buffer_data[i][idx])
                        if new_buffer_trees[i][idx].stack == []: pdb.set_trace()
                    elif idx >= len(new_buffer_log_probs[i]):
                        top_k_trees[i].append( finished_trees[i][idx - num_new_buffer] )
                        top_k_log_probs[i].append( finished_log_probs[i][idx - num_new_buffer] )
            
        result_k_trees, result_k_log_probs, result_k_smiles = [[] for _ in range(len(top_k_trees))], [[] for _ in range(len(top_k_trees))], [[] for _ in range(len(top_k_trees))]
        for i in range(len(top_k_trees)):
            log_prob = top_k_log_probs[i]
            
            sort_prob_key = [k for k, prob in sorted(zip(np.arange(len(log_prob)), log_prob), reverse=True, key=lambda x: x[1])]
            
            result_k_trees[i] = [top_k_trees[i][idx] for idx in sort_prob_key][:num_k]
            result_k_log_probs[i] = [top_k_log_probs[i][idx] for idx in sort_prob_key][:num_k]
            result_k_smiles[i] = [top_k_trees[i][idx].smiles for idx in sort_prob_key]
        
        return result_k_trees, result_k_log_probs
    
    def update_buffer(self, new_tree, new_prob, new_zvec_idx, \
                      new_buffer_data, finished_trees, finished_log_probs, \
                      visited_smiles, trees, tidx, last_vecs=None):
        """ add predicted new molecule to the buffer 
        """
        new_buffer_trees, new_buffer_log_probs, new_buffer_zvec_idxs, new_last_buffer_data = new_buffer_data
        
        if len(new_tree.stack) == 0:
            is_finished = True
            num = len(visited_smiles)
            
            if new_tree.smiles not in visited_smiles:
                finished_trees[tidx].append( new_tree )
                visited_smiles.append( new_tree.smiles )
                   
            if num != len(visited_smiles): 
                finished_log_probs[tidx].append( new_prob )
        else:
            new_buffer_trees[tidx].append( new_tree )
            new_last_buffer_data[tidx].append( last_vecs )
            new_buffer_zvec_idxs[tidx].append( new_zvec_idx )
            new_buffer_log_probs[tidx].append( new_prob )
                
        
    
    
    def decode_topo(self, trees, iter_num, top_k_trees, top_k_log_probs, 
                      top_k_h_vecs, topo_data):
        """ de
        """
        cls_vecs, tree_batches, tree_batch_idxs, tree_zvec_idxs, embedding, atom_vecs, graph_scope = topo_data
        
        new_tree_batches, new_tree_batch_idxs, new_tree_zvec_idxs = [], [], []
        
        tmp_h_vecs, tmp_r_vecs = [], []
        orders = [[] for _ in tree_batches]
        attachatom_idxs = [0 for _ in tree_batches]
        
        for k, tree in enumerate(tree_batches):
            tidx = tree_batch_idxs[k]
            
            zvec_idx = tree_zvec_idxs[k]
            
            graph = tree.mol_graph
            
            atom_id = tree.stack[-1]
            orders[k].append(atom_id)
            
            tmp_h_vecs.append( top_k_h_vecs[zvec_idx][tidx,:] ) 
            if self.use_class: tmp_r_vecs.append(cls_vecs[tidx, :])
        
        stop_atomx_idxs, stop_atomx_idxs1, stop_batch_idxs, stop_zvec_idxs = [], [], [], []
        
        for k, order in enumerate(orders):
            tmp_idx = tree_batch_idxs[k]
            
            scope = graph_scope[k]
            
            for atom_idx in order:
                stop_batch_idxs.append(tree_batch_idxs[k])
                stop_atomx_idxs.append(atom_idx + scope[0])
                stop_atomx_idxs1.append(atom_idx)
                stop_zvec_idxs.append(tree_zvec_idxs[k])
                
        stop_atomx_idxs = torch.LongTensor(stop_atomx_idxs).to(device)
        
        tmp_h_vecs = torch.stack( tmp_h_vecs, dim=0)
        
        # Get the embeddings and ground truth labels for node type predictions
        pre_hatom = index_select_ND(atom_vecs, 0, stop_atomx_idxs)
        topo_hiddens = [pre_hatom]
        if self.update_embed:
            topo_hiddens.append(embedding)

        if self.use_attachatom:
            attachatom_idxs = torch.LongTensor(attachatom_idxs).to(device)
            attachatom_embed = index_select_ND(atom_vecs, 0, attachatom_idxs)
            topo_hiddens.append(attachatom_embed)
                
        topo_hiddens.append( tmp_h_vecs )
        if self.use_class:
            tmp_r_vecs = torch.stack( tmp_r_vecs, dim=0)
            topo_hiddens.append( tmp_r_vecs )
        
        topo_hiddens = torch.cat(topo_hiddens, dim=1)
        topo_scores = self.predict(topo_hiddens, "topo").detach()
        topo_likelihoods = get_likelihood( topo_scores )
        
        stop_trees = [None for tree in trees]
        stop_likelihoods = [0 for tree in trees]
        nonstop_likelihoods = [0 for tree in trees]
        
        for i in range(topo_likelihoods.shape[0]):
            tree_idx = stop_batch_idxs[i]
            tree = trees[tree_idx]
            
            atom_idx = stop_atomx_idxs1[i]
        
            stop_tree = copy.deepcopy(tree)    
            stop_tree.stack = stop_tree.stack[:-1]
            stop_trees[tree_idx] = stop_tree
            
            stop_likelihoods[tree_idx] = topo_likelihoods[i, 1].item()
            nonstop_likelihoods[tree_idx] = topo_likelihoods[i, 0].item()
        
        node_data = (topo_hiddens, stop_atomx_idxs1, stop_batch_idxs, stop_zvec_idxs)
        
        return stop_trees, stop_likelihoods, nonstop_likelihoods, node_data
    

def try_add_mol(vocab, tree, atom_idx, node_label):
    """ determine whether the predicted new node can be attached to the parent node or not.
    """
    smiles = vocab.vocab[node_label]
    mol = get_mol(smiles)
    
    
    attach_atom = tree.mol.GetAtomWithIdx(atom_idx)
    has_atom = False
    match_atom_idx = -1
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == attach_atom.GetSymbol() and atom.GetFormalCharge() == attach_atom.GetFormalCharge():
            # S can have 6 valence and 2 valence
            if check_attach_atom_valence(atom, attach_atom):
                has_atom = True
                match_atom_idx = atom.GetIdx()
                break
    if not has_atom:
        tree.mol_graph = None
        return None
   
     
    graph = tree.mol_graph
    last_graph_size = len(graph)
    attach_mol_graph(graph, mol, [atom_idx], [match_atom_idx])
    try:
        old_mol, map_dict = graph_to_mol( graph, return_map=True)
        
        atom_orders = [-1 for _ in range(old_mol.GetNumAtoms())]
        for key in map_dict:
            atom_orders[key] = map_dict[key]
        mol = Chem.RenumberAtoms(old_mol, atom_orders)
        update_graph_feature(graph, mol, atom_idx)
        
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "P" and atom.GetTotalValence() == 7:
                tree.mol_graph = None
                return None
    except Exception as e:
        tree.mol_graph = None
        return None
    for idx in range(len(graph.nodes)-1, last_graph_size-1, -1):
        if check_atom_valence(mol.GetAtomWithIdx(idx)):
            tree.stack.append(idx)
    
    return mol

def update_graph_feature(graph, mol, atom_idx):
    atom = mol.GetAtomWithIdx(atom_idx)
    
    graph.nodes[atom_idx]['charge'] = atom.GetFormalCharge()
    graph.nodes[atom_idx]['valence'] = atom.GetTotalValence()
    graph.nodes[atom_idx]['aroma'] = atom.GetIsAromatic()
    graph.nodes[atom_idx]['num_h'] = atom.GetTotalNumHs()
    graph.nodes[atom_idx]['atomic_num'] = atom.GetAtomicNum()
    graph.nodes[atom_idx]['in_ring'] = atom.IsInRing()
    
    for edge in graph.edges(atom_idx):
        bond = mol.GetBondBetweenAtoms(edge[0], edge[1])
        graph[edge[0]][edge[1]]['is_conju'] = bond.GetIsConjugated()
        graph[edge[1]][edge[0]]['is_conju'] = bond.GetIsConjugated()   

    
def get_cand_trees(data, vocab=None, num_k=10):
    step, node_likelihood, sort_nodes, tree, atom_idx = data
    
    tmp_trees, tmp_probs = [],[]
     
    for i, node in enumerate(sort_nodes[:num_k]):
        attach_smile = vocab.vocab[node.item()]
        
        to_attach_atom = tree.mol_graph.nodes[atom_idx]['label'][0]
        
        pnum = tree.mol_graph.number_of_nodes()
        if to_attach_atom not in attach_smile:
            continue
        
        cur_idx = len(tree.stack) - 1
        new_tree = copy.deepcopy(tree)
        try:
            mol = try_add_mol( vocab, new_tree, atom_idx, node.item())
        except Exception as e:
            print("error")
            del new_tree
            continue
        
        if new_tree.mol_graph is None:
            del new_tree
            continue
        
        smiles = Chem.MolToSmiles(mol)
        new_tree.mol = mol
        new_tree.smiles = smiles
         
        if not check_atom_valence(new_tree.mol.GetAtomWithIdx(atom_idx)):
            del new_tree.stack[cur_idx]
        
        new_prob = node_likelihood[node].item()
        
        tmp_trees.append(new_tree)
        tmp_probs.append(new_prob)
    
    return tmp_trees, tmp_probs


