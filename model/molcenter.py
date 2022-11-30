import pdb
import random
import time
import copy
import torch
import torch.nn as nn
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Descriptors
from mol_enc import MolEncoder
from mol_tree import MolTree, identify_react_ids
from sklearn.metrics import recall_score
from chemutils import set_atommap, copy_edit_mol, get_ranked_atom_charges, get_smiles, graph_to_mol, get_mol, get_idx_from_mapnum, get_mapnum_from_idx
from config import device, SUB_CHARGE_NUM, SUB_CHARGE_OFFSET, SUB_CHARGE_CHANGE_NUM, BOND_SIZE, \
             VALENCE_NUM, HYDROGEN_NUM, IS_RING_NUM, IS_CONJU_NUM, REACTION_CLS, IS_AROMATIC_NUM
from nnutils import variable_CE_loss, get_likelihood, variable_likelihood, create_pad_tensor, index_select_ND

def make_cuda(tensors, product=True, ggraph=None):
    make_tensor = lambda x: x if type(x) is torch.Tensor else torch.tensor(x, requires_grad=False)
    
    if len(tensors) == 2:
        graph_tensors, tree_tensors = tensors
        new_tree_tensors1 = [x if x is None else make_tensor(x).to(device).long() for x in tree_tensors[:6]]
        new_tree_tensors = new_tree_tensors1 + [tree_tensors[-1]]
    else:
        graph_tensors = tensors[0]
        
    new_graph_tensors1 = [make_tensor(x).to(device).long() for x in graph_tensors[:6]]
    
    if len(graph_tensors) > 8:
        new_graph_tensors2 = [graph_tensors[-3], make_tensor(graph_tensors[-2]).to(device), make_tensor(graph_tensors[-1]).to(device)]
        new_graph_tensors = new_graph_tensors1 + new_graph_tensors2
    elif not product:
        new_graph_tensors = new_graph_tensors1 + [graph_tensors[-1], None, None]
    else:
        new_graph_tensors = new_graph_tensors1 + [graph_tensors[-1]]
    
    if len(tensors) == 2:
        return new_graph_tensors, new_tree_tensors, make_tensor(ggraph).to(device)
    else:
        return [new_graph_tensors]


class MolCenter(nn.Module):
    """ model used to find the reaction center of molecule
    """
    def __init__(self, vocab, avocab, args):
        super(MolCenter, self).__init__()
        self.vocab = vocab
        self.avocab = avocab
        self.hidden_size = args.hidden_size
        self.latent_size = args.latent_size
        self.atom_size = atom_size = avocab.size()
        self.use_brics = args.use_brics
        self.use_feature = args.use_feature
        self.use_tree = args.use_tree
        self.use_latent_attachatom = args.use_latent_attachatom
        self.use_class = args.use_class 
        self.use_mess  = args.use_mess  # whether using the bond features
        
        self.charge_offset = SUB_CHARGE_OFFSET
        self.charge_change_num = SUB_CHARGE_CHANGE_NUM
        self.charge_num = SUB_CHARGE_NUM
        
        # embedding for substructures and atoms
        self.E_a = torch.eye(atom_size).to(device)
        if args.use_feature:
            self.E_fv = torch.eye( VALENCE_NUM ).to(device)
            self.E_fg = torch.eye( self.charge_num ).to(device)
            
            self.E_fh = torch.eye( HYDROGEN_NUM ).to(device)
            self.E_fr = torch.eye( IS_RING_NUM ).to(device)
            self.E_fc = torch.eye( IS_CONJU_NUM ).to(device)
            self.E_fa = torch.eye( IS_AROMATIC_NUM ).to(device)
        
        self.E_b = torch.eye(BOND_SIZE).to(device)
       
        if self.use_feature:
            feature_embedding = (self.E_a, self.E_fv, self.E_fg, self.E_fh, self.E_fr, self.E_fc, self.E_fa, self.E_b)
        else:
            feature_embedding = (self.E_a, self.E_b) 

        if self.use_class:
            self.reactions = torch.eye( REACTION_CLS ).to(device)
            feature_embedding += (self.reactions, )
        
        # number of bond charge types: delete bond; single; double; triple;; 
        self.E_bc = torch.eye(4).to(device)
        
        # encoder
        self.encoder = MolEncoder(self.atom_size, feature_embedding, args=args)
        # weight for atom center function
        self.W_ta = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(self.hidden_size, 1)).to(device)
        
        # weight for bond center function
        self.W_tb = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(self.hidden_size, 4)).to(device)
        
        # weight for bond change function
        self.W_bc = nn.Sequential(nn.Linear(self.hidden_size * 3, self.hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(self.hidden_size, 4)).to(device)
        
        self.W_b = nn.Sequential(nn.Linear(self.hidden_size + 4, self.hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_size, self.hidden_size)).to(device)
        
        # weight for atom charge change function
        self.W_tac = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_size, self.charge_change_num)).to(device)
        
        if self.use_feature and not self.use_mess:
            self.U_t1 = nn.Sequential(nn.Linear(BOND_SIZE + IS_RING_NUM + IS_CONJU_NUM + IS_AROMATIC_NUM + self.hidden_size * 2, self.hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_size, self.hidden_size)).to(device)
        elif not self.use_mess:
            self.U_t1 = nn.Sequential(nn.Linear(BOND_SIZE + self.hidden_size * 2, self.hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_size, self.hidden_size)).to(device)
        
        elif self.use_feature and self.use_mess:
            self.U_t1 = nn.Sequential(nn.Linear(BOND_SIZE + IS_RING_NUM + IS_CONJU_NUM + IS_AROMATIC_NUM + self.hidden_size * 4, self.hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_size, self.hidden_size)).to(device)
        else:
            self.U_t1 = nn.Sequential(nn.Linear(BOND_SIZE + self.hidden_size * 4, self.hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_size, self.hidden_size)).to(device)
            
        self.bond_charge_loss = nn.CrossEntropyLoss()
        self.atom_charge_loss = nn.CrossEntropyLoss()
    
    
    def convert_bonds(self, bond_type_labels, cand_atom1_vecs, cand_atom2_vecs):
        """ get bond embedding from bond features and atom vectors
        """
        sum_atom_vecs = cand_atom1_vecs + cand_atom2_vecs
        diff_atom_vecs = torch.abs(cand_atom1_vecs - cand_atom2_vecs)
        
        if self.use_feature:
            bond_types = self.E_b.index_select(index=bond_type_labels[:,0], dim=0)
            bond_conju = self.E_fc.index_select(index=bond_type_labels[:,1], dim=0)
            bond_ring = self.E_fr.index_select(index=bond_type_labels[:,2], dim=0)
            bond_aroma = self.E_fa.index_select(index=bond_type_labels[:,3], dim=0)
            bond_types = torch.cat( (bond_types, bond_conju, bond_ring, bond_aroma), dim=1)
        else:
            bond_types = self.E_b.index_select(index=bond_type_labels, dim=0)
        
        bond_embeds = self.U_t1( torch.cat( (bond_types, sum_atom_vecs, diff_atom_vecs), dim=1) )
        
        return bond_embeds
        
    def encode(self, tensors, product=False, classes=None, use_feature=False, usemask=False):
        """ Encode the molecule during the test
        
        Args:
            tensors: input embeddings
        """
        tensors[0][0][:, 2] = tensors[0][0][:, 2] + int((self.charge_num - 1 ) / 2)
        mol_rev_vecs, mol_atom_vecs, mol_mess_vecs = self.encoder(tensors, product=product, classes=classes, use_feature=use_feature, usemask=usemask)
        
        return mol_rev_vecs, mol_atom_vecs, mol_mess_vecs
    
    def fuse_noise(self, tree_vecs, mol_vecs):
        """ Add noise to the learned embeding during the testing
        """
        tree_eps = torch.randn(tree_vecs.size(0), self.latent_size).to(device)
        tree_eps = tree_eps.expand(1, tree_vecs.size(1))
         
        mol_eps = torch.randn(mol_vecs.size(0), self.latent_size).to(device)
        mol_eps = mol_eps.expand(1, mol_vecs.size(1))
        return tree_eps, mol_eps
        
    def get_center_embeds(self, product_atom_vecs, cand_bond_types, cand_bond_atom_idxs, cand_atom_atom_idxs):
        """ get the embeddings of all the candidate centers
        """
        # get the embeddings for all the candidate atoms
        cand_atoms_embeds = index_select_ND(product_atom_vecs, 0, cand_atom_atom_idxs)
        
        # get the embeddings for all the candidate bonds
        cand_bond_atom1_vecs = index_select_ND(product_atom_vecs, 0, cand_bond_atom_idxs[:, 0])
        cand_bond_atom2_vecs = index_select_ND(product_atom_vecs, 0, cand_bond_atom_idxs[:, 1])
        cand_bonds_embeds = self.convert_bonds(cand_bond_types, cand_bond_atom1_vecs, cand_bond_atom2_vecs)
        
        #cand_rings_embeds = index_select_ND(product_atom_vecs, 0, cand_ring_atom_idxs).sum(dim=1)
        return cand_atoms_embeds, cand_bonds_embeds #, cand_rings_embeds
        
    def get_bond_change_hiddens(self, bond_change_idxs, bond_center_idxs, cand_bond_embeds, product_embed_vecs):
        """ get the embeddings of bonds for induced bond charge change predictions
        """
        bond_change_idxs = torch.LongTensor( bond_change_idxs ).to(device)
        bond_change_hiddens1 = index_select_ND( cand_bond_embeds, 0, bond_change_idxs )

        bond_center_idxs = torch.LongTensor( bond_center_idxs ).to(device)
        bond_change_hiddens2 = index_select_ND( cand_bond_embeds, 0, bond_center_idxs )
        
        bond_change_hiddens = torch.cat( (bond_change_hiddens1, bond_change_hiddens2, product_embed_vecs), dim=1)
        
        return bond_change_hiddens
    
    def get_atom_charge_hiddens(self, atom_charge_data, cand_center_embeds, product_atom_vecs, has_label=False):
        """ get the embeddings of atoms for atom charge predictions
        """
        atom_charge_idxs = torch.LongTensor([idx for idx, _, _ in atom_charge_data]).to(device)
        batch_center_idxs = torch.LongTensor([idx+1 for _, _, idx in atom_charge_data]).to(device)
        
        
        pad_cand_center_embeds = torch.cat( (torch.zeros( 1, cand_center_embeds.shape[1]).to(device), cand_center_embeds), dim=0)
        
        try: 
            atom_charge_hiddens1 = index_select_ND( product_atom_vecs, 0, atom_charge_idxs )
        except Exception as e:
            print(e)
            raise ValueError("cannot select atom index")
        
        atom_charge_hiddens2 = index_select_ND( pad_cand_center_embeds, 0, batch_center_idxs )
        atom_charge_hiddens = torch.cat( (atom_charge_hiddens1, atom_charge_hiddens2), dim=1)
        
        atom_charge_labels = None
        if has_label: atom_charge_labels = torch.LongTensor([label for _, label, _ in atom_charge_data]).to(device)
        
        return atom_charge_hiddens, atom_charge_labels
        
    def get_atom_charge_hiddens2(self, atom_charge_idxs, cand_bond_embed, product_atom_vecs, has_label=False):
        """ get the embeddings of atoms for atom charge predictions
        """ 
        atom_charge_idxs = torch.LongTensor(atom_charge_idxs).to(device)
        
        atom_charge_hiddens1 = index_select_ND( product_atom_vecs, 0, atom_charge_idxs )
        atom_charge_hiddens2 = cand_bond_embed.unsqueeze(0).repeat(len(atom_charge_idxs), 1)
        atom_charge_hiddens = torch.cat( (atom_charge_hiddens1, atom_charge_hiddens2), dim=1)
        return atom_charge_hiddens
        
    def get_center_logits(self, product_embed_vecs, cand_center_embeds, cand_center_idxs, target, is_bond=False, is_ring=False):
        """ get the logits for center prediction
        """
        select_product_embed_vecs = index_select_ND(product_embed_vecs, 0, cand_center_idxs)
        try:
            center_hiddens = torch.cat( (cand_center_embeds, select_product_embed_vecs), dim=1)
        except:
            pdb.set_trace()
        if is_bond:
            center_logits = self.W_tb( center_hiddens )
            target.index_put_([torch.LongTensor([0, 1, 2, 3]).repeat(len(cand_center_idxs)).to(device), torch.repeat_interleave(cand_center_idxs, 4)], center_logits.flatten())
        elif is_ring:
            center_logits = self.W_tr( center_hiddens )
            target.index_put_([torch.zeros((len(cand_center_idxs)), dtype=int).to(device), cand_center_idxs], center_logits.squeeze(1))
        else:
            center_logits = self.W_ta( center_hiddens )
            target.index_put_([torch.zeros((len(cand_center_idxs)), dtype=int).to(device), cand_center_idxs], center_logits.squeeze(1))
            
    def get_center_hiddens(self, cand_bond_idxs, cand_bond_types, cand_bond_embeds):
        """
        """
        cand_bond_idxs = create_pad_tensor(cand_bond_idxs).to(device)
        cand_bond_types = create_pad_tensor(cand_bond_types).to(device)
        
        pad_cand_bond_embeds = torch.cat( (torch.zeros( 1, cand_bond_embeds.shape[1]).to(device), cand_bond_embeds), dim=0)
        select_bond_embeds = index_select_ND( pad_cand_bond_embeds, 0, cand_bond_idxs)
        select_bond_types = index_select_ND(self.E_bc, 0, cand_bond_types)
        
        select_bond_hiddens = torch.cat( (select_bond_embeds, select_bond_types), dim=2)
        center_charge_hiddens = self.W_b( select_bond_hiddens ).sum(dim=1)
        return center_charge_hiddens
        

    def predict_centers(self, product_bond_tensors, product_embed_vecs, product_atom_vecs, product_mess_vecs, product_trees, product_graph_scopes, product_orders):
        """
        """
        product_hiddens, product_change_hiddens = [], []
        labels, charge_change_labels, bond_change_labels, lengths = [], [], [], []
        cand_bond_types = []
        # data for bond center change
        bond_center_types = []
        # data for next atom charge change prediction
        atom_charge_data = []
        cand_is_bond_idxs, cand_is_atom_idxs, cand_change_bond_idxs, cand_change_center_bond_idxs = [], [], [], []
        cand_bond_atom_idxs, cand_atom_atom_idxs, bond_center_idxs = [], [], []
        
        bonds_offset, cand_bonds_offset, atoms_offset, cand_offset = 1, 0, 1, 0
        for i, tree in enumerate(product_trees):
            bond_order, change_order, ring_order, atom_order = product_orders[i]
            
            bond_size, atom_size = tree.mol_graph.number_of_edges(), tree.mol_graph.number_of_nodes()
            
            bond_tensor = product_bond_tensors[bonds_offset:bonds_offset+bond_size, :]
            one_dir_bond_idxs = torch.where(bond_tensor[:, -1] == 0)[0]
            
            bond_tensor = bond_tensor[one_dir_bond_idxs, :]
            cand_bond_size = bond_tensor.shape[0]
            cand_change_bond_size = 0
            
            if bond_order[0][1] == -1:
                # atom_center
                labels.append(4 * cand_bond_size + bond_order[0][0] - atoms_offset)
            else:
                bond_center_type, bond_center_idx = [], []
                for bond_idx, bond_type in bond_order:
                    bond_idx = torch.where(one_dir_bond_idxs == bond_idx-bonds_offset)[0].item()
                    bond_center_idx.append(bond_idx + cand_bonds_offset + 1)
                    bond_center_type.append(bond_type)
                bond_center_types.append(bond_center_type)
                bond_center_idxs.append(bond_center_idx)
                
                if ring_order is None:
                    # bond_center
                    bond_idx = torch.where(one_dir_bond_idxs == bond_order[0][0]-bonds_offset)[0].item()
                    labels.append( 4 * bond_idx + bond_order[0][1] )
                    
                    # break a bond
                    if bond_order[0][1] == 0:
                        tmp_bond_change_label = [-1, -1]
                        num = 0
                        
                        for j, cand_bond_idx in enumerate(one_dir_bond_idxs):
                            if cand_bond_idx + bonds_offset in change_order[1]:
                                cand_change_bond_idxs.append( cand_bonds_offset + j )
                                num += 1
                            if change_order[0] is not None and cand_bond_idx + bonds_offset == change_order[0][0]:
                                tmp_bond_change_label[0] = num
                                tmp_bond_change_label[1] = change_order[0][1]
                                cand_change_bond_idxs.append(cand_bonds_offset + j)
                                num += 1
                        
                        cand_change_center_bond_idxs.extend( [bond_idx] * num )
                        cand_change_bond_size += num
                        
                        if change_order[0] is None:
                            bond_change_labels.extend([0] * num)
                        else:
                            tmp_bond_change_labels = [0] * num
                            
                            tmp_bond_change_labels[tmp_bond_change_label[0]] = tmp_bond_change_label[1]
                            bond_change_labels.extend( tmp_bond_change_labels )
                            
            
            cand_bond_types.append(bond_tensor[:, 2:])
            cand_bond_atom_idxs.append(bond_tensor[:, :2])
            cand_atom_atom_idxs.extend([j for j in range(atoms_offset, atoms_offset + atom_size)])
            
            for atom_idx, atom_charge_type in atom_order:
                if bond_order[0][1] == -1:
                    atom_charge_data.append( (atom_idx, atom_charge_type+self.charge_offset, -1) )
                else:
                    atom_charge_data.append( (atom_idx, atom_charge_type+self.charge_offset, len(bond_center_idxs)-1) )
                
            cand_is_bond_idxs.extend([j for j in range(cand_offset, cand_offset + cand_bond_size)])
            cand_is_atom_idxs.extend([j for j in range(cand_offset + cand_bond_size, cand_offset + cand_bond_size + atom_size)])
            
            bonds_offset += bond_size
            cand_bonds_offset += cand_bond_size
            atoms_offset += atom_size
            cand_offset  += (cand_bond_size + atom_size)
            lengths.append( (cand_bond_size, atom_size, 4 * cand_bond_size + atom_size) )
            product_embed_vec = product_embed_vecs[i].repeat( (cand_bond_size + atom_size, 1) )
            product_hiddens.append(product_embed_vec)
            product_change_embed_vec = product_embed_vecs[i].repeat( (cand_change_bond_size, 1) )
            product_change_hiddens.append(product_change_embed_vec)
        
        cand_is_bond_idxs = torch.LongTensor(cand_is_bond_idxs).to(device)
        cand_is_atom_idxs = torch.LongTensor(cand_is_atom_idxs).to(device)
        
        cand_bond_atom_idxs = torch.cat(cand_bond_atom_idxs, dim=0).to(device)
        cand_atom_atom_idxs = torch.LongTensor(cand_atom_atom_idxs).to(device)
         
        cand_bond_types = torch.cat(cand_bond_types, dim=0).to(device)
        product_hiddens = torch.cat( product_hiddens, dim=0 )
        product_change_hiddens = torch.cat( product_change_hiddens, dim=0 )
        
        cand_atoms_embeds, cand_bonds_embeds = self.get_center_embeds(product_atom_vecs, cand_bond_types, cand_bond_atom_idxs, cand_atom_atom_idxs)
        cand_logits = torch.zeros( (4, cand_offset) ).to(device)
        
        self.get_center_logits( product_hiddens, cand_atoms_embeds, cand_is_atom_idxs, cand_logits, is_bond=False)
        self.get_center_logits( product_hiddens, cand_bonds_embeds, cand_is_bond_idxs, cand_logits, is_bond=True)
        
        max_length = max([length[2] for length in lengths])
        pad_logits = torch.ones((len(product_graph_scopes), max_length)).to(device) * -100
        
        start = 0
        
        for i, (num_bond, num_atom, length) in enumerate(lengths):
            pad_logits[i, :num_bond*4] = cand_logits[:, start:start + num_bond].transpose(0, 1).flatten()
            pad_logits[i, num_bond*4:num_bond*4+num_atom] = cand_logits[0, start+num_bond:start+num_bond+num_atom]
            start += (num_bond + num_atom)
            
        lengths = torch.LongTensor([length[2] for length in lengths]).to(device)
        
        labels = torch.LongTensor(labels).to(device)
        
        loss = variable_CE_loss(pad_logits, labels, lengths)
        _, center = torch.max(pad_logits, dim=1)
        center_acc = torch.eq(center, labels).float()
        center_acc = torch.sum(center_acc) / labels.nelement()
        # get hiddens for charge prediction
        
        # bond change
        bond_change_hiddens = self.get_bond_change_hiddens(cand_change_bond_idxs, cand_change_center_bond_idxs, cand_bonds_embeds, product_change_hiddens)
        bond_change_labels = torch.LongTensor( bond_change_labels ).to(device)
        
        # atom change
        center_hiddens = self.get_center_hiddens( bond_center_idxs, bond_center_types, cand_bonds_embeds )
        atom_charge_hiddens, atom_charge_labels = self.get_atom_charge_hiddens(atom_charge_data, center_hiddens, product_atom_vecs, has_label=True)
        
        return loss, center_acc.item(), labels.shape[0], (bond_change_hiddens, bond_change_labels), (atom_charge_hiddens, atom_charge_labels)
    
    def validate_centers(self, react_cls, product_batch, product_trees, skip_idxs, has_gt=False, knum=1):
        """
        """
        
        product_graphs, product_tensors, product_ggraph, product_orders = product_batch
        
        product_tensors = make_cuda(product_tensors, ggraph=product_ggraph)
        product_graph_scopes = product_tensors[0][-1]
        
        if self.use_class:
            react_cls = torch.LongTensor(react_cls).to(device)
        else:
            react_cls = None
        # encoding
        
        product_embed_vecs, product_atom_vecs, product_mess_vecs = self.encode(product_tensors, classes=react_cls, use_feature=self.use_feature, product=True)
         
        product_data = (product_embed_vecs, product_tensors, product_atom_vecs, product_trees)
        product_bond_tensors = product_tensors[0][1]
        
        product_hiddens = []
        labels, lengths = [], []
        cand_bond_types = []
        # data for next bond charge change prediction
        bond_elec_types, bond_elec_idxs = [], []
        # data for next atom charge change prediction
        atom_charge_data = []
        cand_is_bond_idxs, cand_is_atom_idxs = [], []
        cand_bond_atom_idxs, cand_atom_atom_idxs = [], []
        bond_center_idxs, atom_center_idxs = [], []
        bonds_offset, atoms_offset, cand_offset = 1, 1, 0
        for i, tree in enumerate(product_trees):
            bond_order, change_order, _, atom_order = product_orders[i]
            
            bond_size, atom_size = tree.mol_graph.number_of_edges(), tree.mol_graph.number_of_nodes()
            
            bond_tensor = product_bond_tensors[bonds_offset:bonds_offset+bond_size, :]
            one_dir_bond_tensor = torch.where(bond_tensor[:, -1] == 0)[0]
            
            bond_tensor = bond_tensor[one_dir_bond_tensor, :]
            cand_bond_size = bond_tensor.shape[0]
            
            if bond_order[0][1] == -1:
                labels.append( 4 * cand_bond_size + bond_order[0][0] - atoms_offset)
                atom_center_idxs.append(i)
            else:
                try:
                    bond_idx = torch.where(one_dir_bond_tensor == bond_order[0][0]-bonds_offset)[0].item()
                except:
                    pdb.set_trace()
                labels.append( 4 * bond_idx + bond_order[0][1] )
                bond_center_idxs.append(i)
            
            cand_bond_types.append(bond_tensor[:, 2:])
            cand_bond_atom_idxs.append(bond_tensor[:, :2])
            cand_atom_atom_idxs.extend([j for j in range(atoms_offset, atoms_offset + atom_size)])
            
            cand_is_bond_idxs.extend([j for j in range(cand_offset, cand_offset + cand_bond_size)])
            cand_is_atom_idxs.extend([j for j in range(cand_offset + cand_bond_size, cand_offset + cand_bond_size + atom_size)])
            
            bonds_offset += bond_size
            atoms_offset += atom_size
            cand_offset  += (cand_bond_size + atom_size)
            lengths.append( (cand_bond_size, atom_size, 4 * cand_bond_size + atom_size) )
            product_embed_vec = product_embed_vecs[i].repeat( (cand_bond_size + atom_size, 1) )
            product_hiddens.append(product_embed_vec)
        
        cand_is_bond_idxs = torch.LongTensor(cand_is_bond_idxs).to(device)
        cand_is_atom_idxs = torch.LongTensor(cand_is_atom_idxs).to(device)
        
        cand_bond_atom_idxs = torch.cat(cand_bond_atom_idxs, dim=0).to(device)
        cand_atom_atom_idxs = torch.LongTensor(cand_atom_atom_idxs).to(device)
        
        cand_bond_types = torch.cat(cand_bond_types, dim=0).to(device)
        
        product_hiddens = torch.cat( product_hiddens, dim=0)
        cand_atoms_embeds, cand_bonds_embeds = self.get_center_embeds(product_atom_vecs, cand_bond_types, cand_bond_atom_idxs, cand_atom_atom_idxs)
        
        cand_logits = torch.empty( (4, len(cand_atom_atom_idxs) + len(cand_bond_atom_idxs)) ).to(device)
        self.get_center_logits( product_hiddens, cand_atoms_embeds, cand_is_atom_idxs, cand_logits, is_bond=False)
        self.get_center_logits( product_hiddens, cand_bonds_embeds, cand_is_bond_idxs, cand_logits, is_bond=True) 
        
        pad_logits = torch.ones((len(product_graph_scopes), max([length[2] for length in lengths]))).to(device) * -100
        
        start = 0
        for i, (num_bond, num_atom, length) in enumerate(lengths):
            pad_logits[i, :num_bond*4] = cand_logits[:, start:start + num_bond].transpose(0, 1).flatten()
            try:
                pad_logits[i, num_bond*4:length] = cand_logits[0, start+num_bond:start+num_bond+num_atom]
            except:
                pdb.set_trace()
            start += (num_bond + num_atom)
        
        lengths = torch.LongTensor([length[2] for length in lengths]).to(device)
        rank_log_probs, log_probs = variable_likelihood(pad_logits, lengths)
        center_ranks = torch.argsort(rank_log_probs, descending=True, dim=1)
        top_k_log_probs = torch.ones( (log_probs.shape[0], log_probs.shape[1]) ).to(device) * -100
         
        bond_center_acc, atom_center_acc = np.zeros( (len(bond_center_idxs), 10) ), np.zeros( (len(atom_center_idxs), 10) )
        top_10_acc = np.zeros( (log_probs.shape[0]+len(skip_idxs), 10) )
        
        for i in range(10):
            for j in range(center_ranks.shape[0]):
                if labels[j] in center_ranks[j, :i+1]:
                    top_10_acc[j, i] = 1
                    if j in bond_center_idxs:
                        idx = bond_center_idxs.index(j)
                        bond_center_acc[idx, i] = 1
                    else:
                        idx = atom_center_idxs.index(j)
                        atom_center_acc[idx, i] = 1
        return top_10_acc, bond_center_acc, atom_center_acc
        
    def test_centers(self, product_embed_vecs, product_atom_vecs, product_trees, product_bond_tensors, product_graph_scopes):
        """
        """
        product_hiddens = []
        labels, lengths = [], []
        cand_bond_types = []
        # data for next bond charge change prediction
        bond_elec_types, bond_elec_idxs = [], []
        # data for next atom charge change prediction
        atom_charge_data = []
        cand_is_bond_idxs, cand_is_atom_idxs = [], []
        cand_bond_atom_idxs, cand_atom_atom_idxs = [], []
        cand_bond_idx_dict = {}
        
        bonds_offset, atoms_offset, cand_offset = 1, 1, 0
        for i, tree in enumerate(product_trees):
            bond_size, atom_size = tree.mol_graph.number_of_edges(), tree.mol_graph.number_of_nodes()
            
            bond_tensor = product_bond_tensors[bonds_offset:bonds_offset+bond_size, :]
            one_dir_bond_tensor = torch.where(bond_tensor[:, -1] == 0)[0]
            for j, val in enumerate(one_dir_bond_tensor): cand_bond_idx_dict[val.item() + bonds_offset] = len(cand_is_bond_idxs) + j
            bond_tensor = bond_tensor[one_dir_bond_tensor, :]
            cand_bond_size = bond_tensor.shape[0]
            
            cand_bond_types.append(bond_tensor[:, 2:])
            cand_bond_atom_idxs.append(bond_tensor[:, :2])
            cand_atom_atom_idxs.extend([j for j in range(atoms_offset, atoms_offset + atom_size)])
            
            lengths.append( (cand_bond_size, atom_size, 4 * cand_bond_size + atom_size, len(cand_is_bond_idxs), len(cand_is_atom_idxs)) )
            
            cand_is_bond_idxs.extend([j for j in range(cand_offset, cand_offset + cand_bond_size)])
            cand_is_atom_idxs.extend([j for j in range(cand_offset + cand_bond_size, cand_offset + cand_bond_size + atom_size)])
             
            bonds_offset += bond_size
            atoms_offset += atom_size
            cand_offset  += (cand_bond_size + atom_size)
            product_embed_vec = product_embed_vecs[i].repeat( (cand_bond_size + atom_size, 1) )
            product_hiddens.append(product_embed_vec)
        
        cand_is_bond_idxs = torch.LongTensor(cand_is_bond_idxs).to(device)
        cand_is_atom_idxs = torch.LongTensor(cand_is_atom_idxs).to(device)
        
        cand_bond_atom_idxs = torch.cat(cand_bond_atom_idxs, dim=0).to(device)
        cand_atom_atom_idxs = torch.LongTensor(cand_atom_atom_idxs).to(device)
        
        cand_bond_types = torch.cat(cand_bond_types, dim=0).to(device)
        
        product_hiddens = torch.cat( product_hiddens, dim=0)
        cand_atoms_embeds, cand_bonds_embeds = self.get_center_embeds(product_atom_vecs, cand_bond_types, cand_bond_atom_idxs, cand_atom_atom_idxs)
        
        cand_logits = torch.empty( (4, len(cand_atom_atom_idxs) + len(cand_bond_atom_idxs)) ).to(device)
        
        self.get_center_logits( product_hiddens, cand_atoms_embeds, cand_is_atom_idxs, cand_logits, is_bond=False)
        self.get_center_logits( product_hiddens, cand_bonds_embeds, cand_is_bond_idxs, cand_logits, is_bond=True) 
         
        pad_logits = torch.ones((len(product_graph_scopes), max([length[2] for length in lengths]))).to(device) * -100
        
        start = 0
        for i, (num_bond, num_atom, length, _, _) in enumerate(lengths):
            pad_logits[i, :num_bond*4] = cand_logits[:, start:start + num_bond].transpose(0, 1).flatten()
            try:
                pad_logits[i, num_bond*4:length] = cand_logits[0, start+num_bond:start+num_bond+num_atom]
            except:
                pdb.set_trace()
            start += (num_bond + num_atom)
        
        cand_lengths = torch.LongTensor([length[2] for length in lengths]).to(device)
        rank_log_probs, log_probs = variable_likelihood(pad_logits, cand_lengths)
        center_ranks = torch.argsort(rank_log_probs, descending=True, dim=1)
        
        top_k_log_probs = -10e5 *  torch.ones( (log_probs.shape[0], log_probs.shape[1]) ).to(device)
        
        for i in range(len(lengths)):
            length = 4 * lengths[i][0] + lengths[i][1]

            
            for j, idx in enumerate(center_ranks[i,:length]):
                top_k_log_probs[i, j] = log_probs[i, idx]
        
        next_charge_data = cand_bond_atom_idxs, cand_atom_atom_idxs, lengths, cand_atoms_embeds, cand_is_atom_idxs, cand_bonds_embeds, cand_is_bond_idxs, cand_bond_idx_dict
        
        return center_ranks, top_k_log_probs, next_charge_data
    
    def get_synthon_padatom_vecs(self, react_tree, synthon_tree, react_atom_vecs, synthon_atom_vecs):
        react_dict = get_mapnum_from_idx(react_tree.mol)
        synthon_dict = get_idx_from_mapnum(synthon_tree.mol)
        
        pad_atom_vecs = torch.zeros_like(react_atom_vecs).to(device)
        for i in range(react_atom_vecs.shape[0]):
            try:
                mapnum = react_dict[i]
            except:
                pdb.set_trace()
            if mapnum == 0: continue
            idx = synthon_dict[mapnum]
            
            pad_atom_vecs[i, :] = synthon_atom_vecs[idx, :]
        return pad_atom_vecs


    def forward(self, classes, product_batch, product_trees):
        """
        Args:
            x_batch: features of molecule X
            y_batch: features of molecule y
            x_trees: list of trees of molecules x
            y_trees: list of trees of molecules y
            beta   : weight for kl loss
        """
        # prepare feature embeddings
        product_graphs, product_tensors, product_ggraph, product_orders = product_batch
        product_tensors = make_cuda(product_tensors, ggraph=product_ggraph)
        
        if self.use_class:
            classes = torch.LongTensor(classes).to(device)
        else:
            classes = None
        # encoding
        product_embed_vecs, product_atom_vecs, product_mess_vecs = self.encode(product_tensors, product=True, use_feature=self.use_feature, classes=classes, usemask=False)
        
        product_data = (product_embed_vecs, product_tensors, product_atom_vecs, product_trees)
        # center prediction
        react_center_loss, center_acc, center_num, bond_charge_data, atom_charge_data = self.predict_centers(product_tensors[0][1], product_embed_vecs, product_atom_vecs, product_mess_vecs, product_trees, product_tensors[0][-1], product_orders)
        
        # bond change prediction
        bond_charge_hiddens, bond_charge_labels = bond_charge_data
        bond_charge_logits = self.W_bc( bond_charge_hiddens )
        try:
            bond_loss = self.bond_charge_loss( bond_charge_logits, bond_charge_labels )
        except:
            pdb.set_trace()
        bond_charge_pred = torch.argmax(bond_charge_logits, dim=1)
        bond_acc = torch.sum(bond_charge_pred == bond_charge_labels).float() / bond_charge_labels.size(0)
        bond_minor_idxs = torch.where(bond_charge_labels != 0)[0]
        
        if bond_minor_idxs.size(0) != 0: bond_rec = torch.sum(bond_charge_pred[bond_minor_idxs] == bond_charge_labels[bond_minor_idxs]).float() / bond_minor_idxs.size(0)
        else: bond_rec = 0
         
        bond_minor_num = torch.sum(bond_charge_labels != 0).item()
        bond_num = bond_charge_labels.shape[0]
        
        # atom charge prediction
        atom_charge_hiddens, atom_charge_labels = atom_charge_data
        atom_charge_logits = self.W_tac( atom_charge_hiddens )
        
        atom_loss = self.atom_charge_loss( atom_charge_logits, atom_charge_labels )
        atom_charge_pred = torch.argmax(atom_charge_logits, dim=1)
        atom_acc = torch.sum(atom_charge_pred == atom_charge_labels).float() / atom_charge_labels.size(0)
        atom_rec = recall_score(atom_charge_labels.cpu(), atom_charge_pred.cpu(), average='macro')
        atom_minor_num = torch.sum(atom_charge_labels != self.charge_offset).item()
        atom_num = atom_charge_labels.shape[0]
        
        # loss
        loss = (react_center_loss, bond_loss, atom_loss)
        acc  = (center_acc, bond_acc.item(), atom_acc.item())
        
        rec  = (bond_rec, atom_rec)
        
        num  = (center_num, bond_minor_num, bond_num, atom_minor_num, atom_num)
        
        total_loss = torch.sum(torch.stack(loss, dim=0))
        
        return total_loss, loss, acc, rec, num
         

    def test_bond_change(self, atom_idxs, center_bond_idx, product_graph, cand_bond_embeds, cand_bond_idx_dict, product_embed, product_tree=None):
        atom1, atom2 = atom_idxs
        
        cand_change_bond_idxs, cand_change_atom_idxs = [], []
        try:
            for bond in list(product_graph.edges(atom1)) + list(product_graph.edges(atom2)):
                if bond[0] == atom1 and bond[1] == atom2: continue
                if bond[0] == atom2 and bond[1] == atom1: continue
                
                catom1, catom2 = bond
                if product_graph[catom1][catom2]['dir'] != 0:
                    catom1, catom2 = catom2, catom1
                
                cand_change_bond_idx = cand_bond_idx_dict[product_graph[catom1][catom2]['mess_idx']]
                cand_change_bond_idxs.append(cand_change_bond_idx)
                cand_change_atom_idxs.append([catom1, catom2])
        except:
            pdb.set_trace()
        
        cuda_cand_change_bond_idxs = torch.LongTensor( cand_change_bond_idxs ).to(device)
        cand_change_bond_embeds = index_select_ND( cand_bond_embeds, 0, cuda_cand_change_bond_idxs )
        
        center_bond_embeds = cand_bond_embeds[center_bond_idx, :].unsqueeze(0).repeat(len(cand_change_bond_idxs), 1)
        product_embeds = product_embed.unsqueeze(0).repeat(len(cand_change_bond_idxs), 1)
        
        cand_change_hiddens = torch.cat( (cand_change_bond_embeds, center_bond_embeds, product_embeds), dim=1 )
        cand_change_logits  = self.W_bc( cand_change_hiddens )
       
        cand_change_likelihoods = get_likelihood(cand_change_logits)
        cand_change_preds = torch.argmax(cand_change_logits, dim=1)
        cand_changed_bonds = torch.where(cand_change_preds != 0)[0]
        
        if len(cand_changed_bonds) != 1:
            return None, 0 
        idx = cand_changed_bonds[0]
        change_bond = cand_change_atom_idxs[idx] + [cand_change_bond_idxs[idx], cand_change_preds[idx].item()]
        return change_bond, cand_change_likelihoods[idx, cand_change_preds[idx]]
        
    def test(self, classes, product_batch, product_trees, react_smiles, decode_type=1, has_gt=False, knum=1):
        product_graphs, product_tensors, product_ggraph, _ = product_batch
        
        product_tensors = make_cuda(product_tensors, ggraph=product_ggraph)
        product_scope = product_tensors[0][-1]
                
        # encoding
        product_embed_vecs, product_atom_vecs, product_mess_vecs = self.encode(product_tensors, product=True, use_feature=self.use_feature, classes=classes, usemask=False)
        
        product_data = (product_embed_vecs, product_tensors, product_atom_vecs, product_trees)
        
        center_ranks, center_log_probs, next_charge_data = self.test_centers(product_embed_vecs, product_atom_vecs, product_trees, product_tensors[0][1], product_tensors[0][-1])
        
        top_k_reacts = [[] for tree in product_trees]
        buffer_log_probs = [[] for tree in product_trees]
        top_k_centers = [[] for tree in product_trees]
        cand_bond_atom_idxs, cand_atom_atom_idxs, length_scope, cand_atom_embeds, cand_is_atoms_idxs, cand_bond_embeds, cand_is_bonds_idxs, cand_bond_idx_dict = next_charge_data
        
        for i in range(knum):
            target_node_idxs = []
            numi_trees = []
            bond_elec_idxs, bond_elec_types, bond_center_idxs, bond_center_types, bond_labels = [], [], [], [], []
            bond_atom_idxs, bond_embed_idxs = [], []
            atom_atom_charge_data, bond_atom_charge_data, bond_atom_charge_idxs, atom_atom_charge_batch_idxs, bond_atom_charge_batch_idxs = [], [], [], [], []
            last_idx = 0
            
            copy_product_trees = copy.deepcopy(product_trees)
            
            for j, tree in enumerate(copy_product_trees):
                if center_ranks[j][i] == -1: continue
                bond_num = length_scope[j][0]
                
                if center_ranks[j][i] < bond_num * 4:
                    center_idx = center_ranks[j][i] // 4 + length_scope[j][3]
                    atom_idx = [idx.item() - length_scope[j][4] -1 for idx in list(cand_bond_atom_idxs[center_idx])]
                else:
                    center_idx = center_ranks[j][i] - 4 * bond_num + length_scope[j][4]
                    atom_idx = [cand_atom_atom_idxs[center_idx].item() - length_scope[j][4] - 1]
                
                if len(atom_idx) == 2:
                    try:
                        tmp = tree.mol_graph[atom_idx[0]][atom_idx[1]]
                    except:
                        pdb.set_trace()
                
                # is atom
                if len(atom_idx) == 1:
                    aidx = atom_idx[0]
                    
                    atom_embed_idx = aidx + length_scope[j][4] + 1
                    atom_atom_charge_data.append( (atom_embed_idx, 0, -1) )
                    atom_atom_charge_batch_idxs.append(j)
                else:
                    aidx1, aidx2 = atom_idx
                    
                    bond_elec_idx = center_ranks[j][i] // 4 + length_scope[j][3]
                    bond_elec_type = center_ranks[j][i] % 4
                    
                    bond_elec_idxs.append(bond_elec_idx)
                    bond_elec_types.append(  (j, -1) )
                    
                    bond_change, bond_change_logits = None, 0
                    if bond_elec_type == 0:
                        
                        bond_change, bond_change_logits = self.test_bond_change([tmp+length_scope[j][4]+1 for tmp in atom_idx], bond_elec_idx, \
                                                                                product_graphs[0], cand_bond_embeds, \
                                                                                cand_bond_idx_dict, product_embed_vecs[j], product_tree=product_trees[j])
                        
                        center_log_probs[j, i] += bond_change_logits
                    
                    bond_center_idx, bond_center_type = [bond_elec_idx], [bond_elec_type]
                    bond_atom_offset, bond_atom_num = len(bond_atom_charge_data), 2
                    bond_label = [(aidx1, aidx2, bond_elec_type)]
                    bond_atom_idx = [tmp + length_scope[j][4] + 1 for tmp in [aidx1, aidx2]]
                    
                    if bond_change is not None:
                        caidx1, caidx2, bond_change_idx, bond_change_type = bond_change
                        bond_atom_num += 2
                        bond_center_idx.append( bond_change_idx )
                        bond_center_type.append( bond_change_type )
                        bond_atom_idx.extend( [caidx1, caidx2] )
                        bond_label.append( (caidx1-length_scope[j][4]-1, caidx2-length_scope[j][4]-1, bond_change_type) )
                    
                    
                    bond_center_idxs.append([bidx + 1 for bidx in bond_center_idx])
                    bond_center_types.append(bond_center_type)
                    bond_atom_idxs.append(bond_atom_idx)
                    
                    bond_labels.append( bond_label )
                    
            atom_center_embeds = torch.zeros( (len(atom_atom_charge_data), self.hidden_size) ).to(device)
            atom_atom_charge_hiddens, _ = self.get_atom_charge_hiddens(atom_atom_charge_data, atom_center_embeds, product_atom_vecs, has_label=False)
            atom_atom_charge_logits = self.W_tac( atom_atom_charge_hiddens )
            atom_atom_charge_likelihoods = get_likelihood(atom_atom_charge_logits)
            atom_atom_charge_ranks = torch.argsort(atom_atom_charge_logits, descending=True, dim=1)
            
            bond_atom_center_hiddens = self.get_center_hiddens( bond_center_idxs, bond_center_types, cand_bond_embeds )
            bond_elec_dicts = {tmp[0]: j for j, tmp in enumerate(bond_elec_types)}
            
            for j, old_tree in enumerate(copy_product_trees):
                if center_ranks[j][i] == -1: continue
                tmp_atom_charge_data = []
                
                if j in bond_elec_dicts:
                    idx = bond_elec_dicts[j]
                    bond_center_atom_idx = bond_atom_idxs[idx]
                    bond_center_hidden = bond_atom_center_hiddens[idx]
                    
                    for tidx1, tidx2, label in bond_labels[idx]:
                        if label == 0:
                            old_tree.mol_graph[tidx1][tidx2]['new_label'] = None
                            old_tree.mol_graph[tidx2][tidx1]['new_label'] = None
                        else:
                            old_tree.mol_graph[tidx1][tidx2]['new_label'] = label - 1
                            old_tree.mol_graph[tidx2][tidx1]['new_label'] = label - 1
                    
                    atom_idx    = [tidx-length_scope[j][4]-1 for tidx in bond_center_atom_idx]
                    atom_labels = [old_tree.mol_graph.nodes[tidx]['charge'] for tidx in atom_idx]
                    # predict atom charge change
                    atom_charge_hiddens = self.get_atom_charge_hiddens2(bond_center_atom_idx, bond_center_hidden, product_atom_vecs, has_label=False)
                    atom_charge_logits = self.W_tac(atom_charge_hiddens)
                    
                    atom_charge_likelihoods = get_likelihood(atom_charge_logits)
                    
                    all_combs, all_log_likelihoods = [], []
                    count = 0
                    
                    atom_charge_ranks = get_ranked_atom_charges(atom_charge_likelihoods)
                    
                    for m, (charges, likelihood) in enumerate(atom_charge_ranks[:2]):
                        all_log_likelihoods.append( (count, likelihood + center_log_probs[j, i].item()) )
                        all_combs.append( charges )
                        count += 1
                    
                    all_log_likelihoods.sort(key=lambda x: x[1], reverse=True)
                    
                    changed_atom_idxs = []
                    add_tree = False
                    for k, likelihood in all_log_likelihoods[:2]:
                        charges = all_combs[k]
                        
                        wrong_atom = False
                        changed_atom_idxs = []
                        
                        for n, charge in enumerate(charges):
                            if charge == self.charge_offset: continue
                            # lose one charge
                            new_atom_charge = atom_labels[n] - (charge - self.charge_offset)
                            
                            if new_atom_charge < -self.charge_offset or new_atom_charge > self.charge_offset: continue
                            
                            old_tree.mol_graph.nodes[atom_idx[n]]['new_charge'] = new_atom_charge if type(new_atom_charge) is int else new_atom_charge.item()
                            changed_atom_idxs.append(atom_idx[n])
                        
                        try:
                            mols = graph_to_mol(old_tree.mol_graph)
                            reacts = get_smiles(mols)
                        except:
                            # charge is not allowed, delete new charge.
                            for aidx in changed_atom_idxs:
                                if 'new_charge' in old_tree.mol_graph.nodes[aidx]:
                                    del old_tree.mol_graph.nodes[aidx]['new_charge']
                            continue
                        
                        top_k_reacts[j].append( (reacts, mols) )
                        buffer_log_probs[j].append( likelihood )
                        top_k_centers[j].append( atom_idx )
                        break
                    
                    
                    for bond in old_tree.mol_graph.edges:
                        if 'new_label' in old_tree.mol_graph[bond[0]][bond[1]]:
                            del old_tree.mol_graph[bond[0]][bond[1]]['new_label']
                            del old_tree.mol_graph[bond[1]][bond[0]]['new_label']
                    
                    for node in old_tree.mol_graph.nodes:
                        if 'new_charge' in old_tree.mol_graph.nodes[node]:
                            del old_tree.mol_graph.nodes[node]['new_charge']
                else:
                    idx = atom_atom_charge_batch_idxs.index(j)
                    charge_ranks = atom_atom_charge_ranks[idx]
                    
                    try:
                        aidx = atom_atom_charge_data[idx][0] - length_scope[j][4] - 1
                    except:
                        pdb.set_trace()
                    
                    changed_atom_idxs = []
                    for m, charge in enumerate(charge_ranks[:2]):
                        if charge != self.charge_offset:
                            changed_atom_idxs = []
                            atom_label = old_tree.mol_graph.nodes[aidx]['charge']
                            
                            # lose one charge
                            new_atom_charge = atom_label + (self.charge_offset - charge)
                            
                            if new_atom_charge > self.charge_offset or new_atom_charge < -self.charge_offset: continue
                            
                            old_tree.mol_graph.nodes[aidx]['new_charge'] = new_atom_charge if type(new_atom_charge) is int else new_atom_charge.item()
                            changed_atom_idxs.append(aidx)
                        
                        likelihood = atom_atom_charge_likelihoods[idx, charge].item() + center_log_probs[j, i].item()
                        
                        try:
                            mols = graph_to_mol(old_tree.mol_graph)
                            reacts = get_smiles(mols)
                        except Exception as e:
                            print(e)
                            
                            if aidx < len(old_tree.mol_graph.nodes) and 'new_charge' in old_tree.mol_graph.nodes[aidx]:
                                del old_tree.mol_graph.nodes[aidx]['new_charge']
                            
                            changed_atom_idxs = []
                            continue
                        
    
                        top_k_reacts[j].append( (reacts, mols) )
                        buffer_log_probs[j].append(likelihood)
                        top_k_centers[j].append( (aidx,) )
                        break
                    
                    
                    if aidx <len(old_tree.mol_graph.nodes) and 'new_charge' in old_tree.mol_graph.nodes[aidx]:
                        del old_tree.mol_graph.nodes[aidx]['new_charge']
        
        for i in range(len(top_k_reacts)):
            log_probs = buffer_log_probs[i]
            
            tmp_buffer_log_probs = [(idx, log_prob) for idx, log_prob in enumerate(log_probs)]
            tmp_buffer_log_probs.sort(key=lambda x:x[1], reverse=True)
            buffer_log_probs[i] = [log_prob for _, log_prob in tmp_buffer_log_probs[:knum]]
            
            new_reacts = []
            new_centers = []
            
            for idx, _ in tmp_buffer_log_probs[:knum]:
                new_reacts.append( top_k_reacts[i][idx] )
                new_centers.append( top_k_centers[i][idx] )
            
            top_k_reacts[i] = new_reacts
            top_k_centers[i] = new_centers
        
        top_k_react_data = [None for _ in range(knum)]
        top_k_synthon_trees = [None for _ in range(knum)]
                
        for i in range(knum):
            reacts_trees = [None for _ in range(len(product_trees))]
            react_id_edges = []
            skip_mols = []
            
            for j, reacts in enumerate(top_k_reacts):
                if len(reacts) <= i: continue
                
                react_atommap = {}
                react_trees = []
                
                prod_idx_to_mapnum = get_mapnum_from_idx(product_trees[j].mol)
                
                try:        
                    react_tree = MolTree(reacts[i][0], mol=reacts[i][1])
                except Exception as e:
                    print(e)
                    print("fail to construct tree!!!!!!!! react: %s" % (reacts[i][0]))
                    reacts_trees[j] = None
                    buffer_log_probs[j] = -100
                    continue
                
                react_tree.finished = False
                react_tree.frag = []
                react_tree.stack = []
                
                mapnum_to_idx = get_idx_from_mapnum(react_tree.mol)
                try:
                    for idx in top_k_centers[j][i]:
                        mapnum = prod_idx_to_mapnum[idx]
                        if mapnum in mapnum_to_idx:
                            atom_id = mapnum_to_idx[mapnum]
                            react_tree.stack.append(atom_id)
                except:
                    pdb.set_trace()
                reacts_trees[j] = react_tree
            
            top_k_synthon_trees[i] = reacts_trees
        
        return top_k_synthon_trees, buffer_log_probs
