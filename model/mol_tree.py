import pdb
import os
import pickle
import rdkit
import argparse
import rdkit.Chem as Chem
import networkx as nx
from rdkit.Chem import Descriptors
from chemutils import graph_to_mol, brics_decomp, get_clique_mol, tree_decomp, get_mol, get_smiles, set_atommap, BOND_LIST, get_bonds_atommap, get_idx_from_mapnum
from nnutils import create_pad_tensor
from vocab import *
import torch
import random

class MolTree(object):
    
    def __init__(self, smiles, mol=None, use_brics=False, isTest=False, decompose_ring=False):
        self.smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        self.mol = get_mol(self.smiles)
        
        if mol is not None and self.mol is None:
            self.mol = mol
        
        self.mol_graph = self.build_mol_graph()
        self.use_brics = use_brics
        
        if use_brics:
            self.brics_cliques, self.brics_edges = brics_decomp(self.mol)
            self.brics_tree = self.build_brics_tree()
        
        self.cliques, self.edges = tree_decomp(self.mol, decompose_ring=decompose_ring)
        self.mol_tree = self.build_mol_tree()
           
        self.order = []
        self.set_anchor()

        
    def build_mol_graph(self):
        mol = self.mol
        
        graph = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(mol))
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            graph.nodes[idx]['label'] = atom.GetSymbol()
            graph.nodes[idx]['idx'] = atom.GetAtomMapNum()
            if graph.nodes[idx]['idx'] == 0: graph.nodes[idx]['edit'] = 1
            else: graph.nodes[idx]['edit'] = 0
             
            graph.nodes[idx]['charge'] = atom.GetFormalCharge()
            graph.nodes[idx]['valence'] = atom.GetTotalValence()
            graph.nodes[idx]['num_h'] = atom.GetTotalNumHs()
            graph.nodes[idx]['aroma'] = atom.GetIsAromatic()
            graph.nodes[idx]['atomic_num'] = atom.GetAtomicNum()
            graph.nodes[idx]['in_ring'] = atom.IsInRing()
            
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = BOND_LIST.index( bond.GetBondType() )
            graph[a1][a2]['label'] = btype
            graph[a2][a1]['label'] = btype
            graph[a1][a2]['dir'] = 0
            graph[a2][a1]['dir'] = 1
            graph[a1][a2]['is_conju'] = bond.GetIsConjugated()
            graph[a2][a1]['is_conju'] = bond.GetIsConjugated()
            graph[a1][a2]['is_aroma'] = bond.GetIsAromatic()
            graph[a2][a1]['is_aroma'] = bond.GetIsAromatic()
            graph[a1][a2]['in_ring'] = bond.IsInRing()
            graph[a2][a1]['in_ring'] = bond.IsInRing()
                            
        return graph
        
    def build_mol_tree(self):
        cliques = self.cliques
        graph = nx.DiGraph()
        
        ring_size = 0
        for i, clique in enumerate(cliques):
            cmol = get_clique_mol(self.mol, clique)
            graph.add_node(i)
            ## tmp =========================
            #is_revise = False
            #for atom in cmol.GetAtoms():
            #    if atom.GetAtomMapNum() == 0:
            #        is_revise = True
            #        break
            #
            #if cmol.GetNumAtoms() > 1 and is_revise: graph.nodes[i]['revise'] = 1
            #else: graph.nodes[i]['revise'] = 0
            ## =============================

            set_atommap(cmol)
            graph.nodes[i]['label'] = get_smiles(cmol)
            graph.nodes[i]['clq'] = clique
            if len(clique) > 2: ring_size += 1
        self.ring_size = ring_size
        
        for edge in self.edges:
            inter_atoms = list(set(cliques[edge[0]]) & set(cliques[edge[1]]))
            
            graph.add_edge(edge[0], edge[1])
            graph.add_edge(edge[1], edge[0])
            graph[edge[0]][edge[1]]['anchor'] = inter_atoms
            graph[edge[1]][edge[0]]['anchor'] = inter_atoms
            
            if len(inter_atoms) == 1:
                graph[edge[0]][edge[1]]['label'] = cliques[edge[0]].index(inter_atoms[0])
                graph[edge[1]][edge[0]]['label'] = cliques[edge[1]].index(inter_atoms[0])
            elif len(inter_atoms) == 2:
                index1 = cliques[edge[0]].index(inter_atoms[0])
                index2 = cliques[edge[0]].index(inter_atoms[1])
                if index2 == len(cliques[edge[0]])-1:
                    index2 = -1
                graph[edge[0]][edge[1]]['label'] = max(index1, index2)
                
                index1 = cliques[edge[1]].index(inter_atoms[0])
                index2 = cliques[edge[1]].index(inter_atoms[1])
                if index2 == len(cliques[edge[1]])-1:
                    index2 = -1
                graph[edge[1]][edge[0]]['label'] = max(index1, index2)
                
        return graph
   
    def build_brics_tree(self):
        cliques = self.brics_cliques
        graph = nx.DiGraph()
        
        for i, clique in enumerate(cliques):
            try:
                cmol = get_clique_mol(self.mol, clique)
            except:
                pdb.set_trace()
            graph.add_node(i)
            set_atommap(cmol)
            graph.nodes[i]['label'] = get_smiles(cmol)
            graph.nodes[i]['clq'] = clique
            for atom in clique:
                self.mol_graph.nodes[atom]['brics_node_idx'] = i
        
        for edge in self.brics_edges:
            graph.add_edge(edge[0], edge[1])
            graph.add_edge(edge[1], edge[0])
            
            if 'anchor' not in graph[edge[0]][edge[1]]:
                graph[edge[0]][edge[1]]['anchor'] = []
                graph[edge[1]][edge[0]]['anchor'] = []

            
            graph[edge[0]][edge[1]]['anchor'].append(edge[2])
            graph[edge[1]][edge[0]]['anchor'].append(edge[3])

            #if len(graph[edge[0]][edge[1]]['anchor']) > 1: pdb.set_trace()
        
        return graph
    
    def set_anchor(self):
        for atom in self.mol_graph.nodes:
            self.mol_graph.nodes[atom]['node_idx'] = []

        for i, clique in enumerate(self.cliques):
            self.mol_tree.nodes[i]['bonds'] = []
            for atom in clique:
                self.mol_graph.nodes[atom]['node_idx'].append(i)
        
        for bond in self.mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()

            for i, clique in enumerate(self.cliques):
                if begin_idx not in clique or end_idx not in clique:
                    continue
                else:
                    if 'node_idx' not in self.mol_graph[begin_idx][end_idx]:
                        self.mol_graph[begin_idx][end_idx]['node_idx'] = []
                        self.mol_graph[end_idx][begin_idx]['node_idx'] = []

                   
                    self.mol_graph[begin_idx][end_idx]['node_idx'].append(i)
                    self.mol_graph[end_idx][begin_idx]['node_idx'].append(i)
                    
                    if self.mol_graph[begin_idx][end_idx]['dir'] == 0:                    
                        self.mol_tree.nodes[i]['bonds'].append([ begin_idx, end_idx])
                    else:
                        self.mol_tree.nodes[i]['bonds'].append([ end_idx, begin_idx])
    
    def set_revise(self, revise_bonds, product=False, shuffle=False, use_dfs=True, charge_change_atoms=[], connect_atoms=[]):
        tree = self.mol_tree
        graph = self.mol_graph
        
        def dfs(order, visited, parent):
            atom, node = parent
            
            new_atoms = [a for a in sorted(self.mol_tree.nodes[node]['clq']) if a != atom]
            if shuffle: random.shuffle(new_atoms)
            
            for new_atom in new_atoms:
                node_idxs = self.mol_graph.nodes[new_atom]['node_idx']
                diff_idxs = set(node_idxs) - visited
                    
                if len(diff_idxs) == 0:
                    order.append((new_atom, -1))
                else:
                    for idx in diff_idxs:
                        order.append((new_atom, idx))
                        visited.add(idx)
                        dfs(order, visited, (new_atom, idx))
                    order.append((new_atom, -1))
                #new_nodes.extend(sorted_child)
            
        def bfs(order, visited, parents):
            if shuffle: random.shuffle(parents)
            
            new_parents = []
            for atom, node in parents:
                new_atoms = [a for a in sorted(self.mol_tree.nodes[node]['clq']) if a != atom]
                if shuffle: random.shuffle(new_atoms)
                
                for new_atom in new_atoms:
                    node_idxs = self.mol_graph.nodes[new_atom]['node_idx']
                    diff_idxs = set(node_idxs) - visited
                        
                    if len(diff_idxs) == 0:
                        order.append((new_atom, -1))
                    else:
                        for node_idx in diff_idxs:
                            order.append((new_atom, node_idx))
                            new_parents.append((new_atom, node_idx))
                            visited.add(node_idx)
                        order.append((new_atom, -1))
            
            if len(new_parents) > 0: bfs(order, visited, new_parents)
        
        unrevise_nodes = set()
        self.revise_bonds = {revise_bond[0]: revise_bond[1] for revise_bond in revise_bonds}
        
        target_idx = -1
        order = []
        
        for i, cls in enumerate(self.cliques):
             
            for revise_bond, bond_type in revise_bonds:
                if revise_bond[0] in cls and revise_bond[1] in cls:
                    if bond_type is None:
                        tree.nodes[i]['revise'] = 1
                        graph[revise_bond[0]][revise_bond[1]]['revise'] = 1
                        graph[revise_bond[1]][revise_bond[0]]['revise'] = 1
                    elif bond_type == 0:
                        target_idx = i
                        tree.nodes[i]['delete'] = 1
                        graph[revise_bond[0]][revise_bond[1]]['delete'] = 1
                        graph[revise_bond[1]][revise_bond[0]]['delete'] = 1
                        
                        if graph[revise_bond[0]][revise_bond[1]]['dir'] == 0:
                            order.append((revise_bond[0], revise_bond[1], 0))
                    elif bond_type > 0:
                        #pdb.set_trace()
                        tree.nodes[i]['change'] = 1
                        graph[revise_bond[0]][revise_bond[1]]['change'] = bond_type
                        graph[revise_bond[1]][revise_bond[0]]['change'] = bond_type
                        
                        if graph[revise_bond[0]][revise_bond[1]]['dir'] == 0:
                            order.append((revise_bond[0], revise_bond[1], bond_type))
                    
            if 'revise' not in tree.nodes[i] and not product: 
                tree.nodes[i]['revise'] = 0
                unrevise_nodes.add(i)
        
        if not product:
            visited = unrevise_nodes
            if shuffle: random.shuffle(connect_atoms)
            new_parents = []
            for atom in connect_atoms:
                nodes = []
                node_idxs = set(self.mol_graph.nodes[atom]['node_idx'])
                if not shuffle: diff_idxs = sorted(list(node_idxs - visited))
                else:
                    diff_idxs = list(node_idxs - visited)
                    random.shuffle(diff_idxs)
                    
                self.mol_graph.nodes[atom]['attach'] = 1  
                if len(diff_idxs) == 0:
                    order.append((atom, -1))
                else:
                    for idx in diff_idxs:
                        order.append((atom, idx))
                        visited.add(idx)
                        new_parents.append((atom, idx))
                        if use_dfs: dfs(order, visited, (atom, idx))
                    
                    order.append((atom, -1))
            if not use_dfs: bfs(order, visited, new_parents)
            
        elif product:
            order = sorted(order, key=lambda x: x[2])
            if len(charge_change_atoms) > 0:
                #pdb.set_trace()
                for atom, product_charge, react_charge in charge_change_atoms:
                    graph.nodes[atom]['charge_change'] = product_charge - react_charge
            
            if len(connect_atoms) > 0:
                for atom in connect_atoms:
                    graph.nodes[atom]['attach'] = 1
                    if len(order) == 0 or order[0][1] == -1:
                        order.append((atom, -1, -1))
                    
            if target_idx >= 0:
                if len(tree.nodes[target_idx]['clq']) > 2:
                    bonds = tree.nodes[target_idx]['bonds']
                    for bond in bonds:
                        if 'delete' not in graph[bond[0]][bond[1]]:
                            graph[bond[0]][bond[1]]['delete'] = 0
                            graph[bond[1]][bond[0]]['delete'] = 0
                
                atoms = list(set([atom for bond in self.revise_bonds for atom in bond]))
                neighbor_bonds = list(graph.edges(atoms))
            
            change, ring = [[], []], [[], [], None]
            if len(order) == 2 and order[0][-1] == 0 and order[1][-1] > 0:
                # delete a bond and change a bond
                change[0] = order[1]
            elif len(order) >= 2:
                node_idxs = [node_idx for tmp in order for node_idx in graph[tmp[0]][tmp[1]]['node_idx']]
                #print(order)
                #print(self.smiles)
                # break a ring
                #if len(set(node_idxs)) == 1:
                #    self.mol_tree.nodes[node_idxs[0]]['break'] = True
                #    ring[0] = self.mol_tree.nodes[node_idxs[0]]['clq']
                #    ring_broken_bonds = [bond_id for bond_id, bond in enumerate(ring[1]) if bond + (0,) in order]
                #    ring[1:] = get_comb_bonds(self.mol_tree.nodes[node_idxs[0]]['bonds'], label=ring_broken_bonds)
                #else:
                raise ValueError("fail to catch")

            if (len(order) == 1 and order[0][-1] == 0) or (len(order) == 2 and order[0][-1] == 0 and len(change[0]) > 0):
                atom_idxs = list(set([atom_idx for tmp in order for atom_idx in tmp[:2]]))
                neighbor_bonds = [edge for atom_idx in atom_idxs for edge in graph.edges(atom_idx) \
                                                 if edge[0] not in atom_idxs or edge[1] not in atom_idxs]
                
                for i, bond in enumerate(neighbor_bonds):
                    if graph[bond[0]][bond[1]]['dir'] != 0:
                        neighbor_bonds[i] = (bond[1], bond[0])
                change[1] = neighbor_bonds
                


            self.change = change
            self.ring = ring
             
        self.order = order
        
    def set_center(self, node_idx):
        tree = self.mol_tree
        graph = self.mol_graph

        tree.nodes[node_idx]['delete'] = 1
        for begin, end in tree.nodes[node_idx]['bonds']:
            graph[begin][end]['delete'] = 0

    def set_deletes(self, bond_idxs):
        graph = self.mol_graph
        
        for bond_idx1, bond_idx2 in bond_idxs:
            graph[bond_idx1][bond_idx2]['delete'] = 1
            graph[bond_idx2][bond_idx1]['delete'] = 1
            if 'delete' not in graph.nodes[bond_idx1]: graph.nodes[bond_idx1]['delete'] = 1
            if 'delete' not in graph.nodes[bond_idx2]: graph.nodes[bond_idx2]['delete'] = 1
            
            bonds = graph.edges(bond_idx1)
            for idx1, idx2 in bonds:
                if (idx1 == bond_idx1 or idx1 == bond_idx2) and \
                   (idx2 == bond_idx1 or idx2 == bond_idx2): continue
                if 'delete' in graph[idx1][idx2] and graph[idx1][idx2]['delete'] == 1: continue
                
                graph[idx1][idx2]['change'] = -1
        
    def set_change(self, bond_idxs):
        graph = self.mol_graph
        for (bond_idx1, bond_idx2), label in bond_idxs:
            graph[bond_idx1][bond_idx2]['change'] = label
            graph[bond_idx2][bond_idx1]['change'] = label    

    def set_attach(self, delete_bond_nums, change_bond_nums):
        mapnum_idx_dict = get_idx_from_mapnum(self.mol)
        atom_idxs = set([mapnum_idx_dict[atom] for bond in delete_bond_nums + change_bond_nums for atom in bond if atom in mapnum_idx_dict])
        
        #if len(atom_idxs) == 0:
        #    for atom_idx in self.mol_graph:
        #        atom = self.mol.GetAtomWithIdx(atom_idx)
        #        if atom.GetTotalNumHs() > 0:
        #            self.mol_graph.nodes[atom_idx]['attach'] = 0
        #else:
        #    for atom_idx in atom_idxs:
        #        self.mol_graph.nodes[atom_idx]['attach'] = 0

    def set_react_ids(self, react_id_edges):
        tree = self.mol_tree
        
        for node in tree.nodes:
            for bond in tree.nodes[node]['bonds']:
                bond = tuple(bond)
                if bond in react_id_edges:
                    if 'syn_id' not in tree.nodes[node]: tree.nodes[node]['syn_id'] = set()
                    try:
                        tree.nodes[node]['syn_id'].update(set(react_id_edges[bond]))
                    except:
                        pdb.set_trace()
                    
    @staticmethod
    def tensorize(mol_batch, vocab, avocab, skip_mols=[], use_atomic=False, use_feature=False, use_brics=False, product=False, istest=False, usemask=True):
        del_num = 0
        for i in range(len(mol_batch)):
            mols = mol_batch[i]
            if mols is None or i in skip_mols: continue
             
            if type(mols) is not list: mols = [mols]
             
        mol_batch = [mol_batch[i] for i in range(len(mol_batch)) if i not in skip_mols]
        
        if product:
            return mol_batch, MolTree.__tensorize(mol_batch, vocab, avocab, product=True, istest=istest, use_atomic=use_atomic, use_brics=use_brics, use_feature=use_feature)
        else:
            react_graphs, react_tensors, _, react_orders = MolTree.__tensorize(mol_batch, vocab, avocab, use_atomic=use_atomic, use_brics=False, use_feature=use_feature, istest=istest) 
            
            if not istest and usemask:
                react_graphs, react_tensors = MolTree.__append_mask(react_graphs[0], react_tensors[0], react_orders)
            
            return mol_batch, (react_graphs, react_tensors, react_orders)

    @staticmethod
    def __tensorize(mol_batch, vocab, avocab, product=False, use_atomic=False, use_brics=False, use_feature=False, add_target=False, istest=False):
        #if not istest:
        graph_tensors, graph_batchG = MolTree.tensorize_graph([(x.mol_graph, x.mol) if x is not None else (None, None) for x in mol_batch], avocab, use_atomic=use_atomic, use_brics=use_brics, tree=False, use_feature=use_feature)
        graph_scope = graph_tensors[-1]
        
        tree_tensors, tree_batchG = None, None
        
        if product and use_brics:
            tree_tensors, tree_batchG = MolTree.tensorize_graph([(x.brics_tree, x.mol) if x is not None else (None, None) for x in mol_batch], avocab, tree=True, use_feature=use_feature)
        
        if graph_batchG is None:
            if use_brics:
                return (graph_batchG, tree_batchG), (graph_tensors, tree_tensors), ([]), ([])
            else:
                return [graph_batchG], [graph_tensors], ([]), ([])
        
        egraph = [[]]
        all_orders = []
        
        if product:
            if use_brics:
                cgraph = [[0]]
                max_cls_size = max( [len(c) for x in mol_batch if x is not None for c in x.brics_cliques] )
                dgraph = torch.zeros(len(tree_batchG) + 1, max_cls_size).long()
                for u,v,attr in tree_batchG.edges(data=True):
                    eid = attr['mess_idx']
                    anchor = tree_batchG[u][v]['anchor']
                    cgraph.append(anchor)
                
                for v,attr in tree_batchG.nodes(data=True):
                    bid = attr['batch_id']
                    offset = graph_scope[bid][0]
                    tree_batchG.nodes[v]['clq'] = cls = [x + offset for x in attr['clq']]
                    #tree_batchG.nodes[v]['bonds'] = [(x + offset, y+offset) for x, y in attr['bonds']]
                    dgraph[v, :len(cls)] = torch.LongTensor(cls)
                tree_tensors = tree_tensors[:4] + (cgraph, dgraph, tree_tensors[-1])
                
                ggraph = torch.zeros(len(graph_batchG)+1, 1).long()
                for u, attr in graph_batchG.nodes(data=True):
                    ggraph[u] = attr['brics_node_idx']
            
            break_rings = {}
            for bid, hmol in enumerate(mol_batch):
                for node_idx in hmol.mol_tree.nodes:
                    node = hmol.mol_tree.nodes[node_idx]
                    if len(node['clq']) > 2:
                        egraph.append([bid] + [x + graph_scope[bid][0] for x in node['clq']])
                    if not istest and 'break' in node: break_rings[bid] = len(egraph) - 1
                    
            if not istest:
                for i,hmol in enumerate(mol_batch):
                    atom_offset = graph_scope[i][0]
                    if hmol is None:
                        all_orders.append([])
                    else:
                        order = [[], [None, []], None, []]
                        
                        if i not in break_rings:
                            # single bond changes
                            for x, y, bond_type in hmol.order:
                                if y == -1:
                                    order[0].append( (x + atom_offset, -1) )
                                elif hmol.mol_graph[x][y]['dir'] == 0:
                                    order[0].append( (graph_batchG[x + atom_offset][y + atom_offset]['mess_idx'], bond_type) )
                                else:
                                    pdb.set_trace()
                                        
                            # additional bond type changes
                            if len(hmol.change[0]) > 0:   
                                change_x, change_y, change_type = hmol.change[0]
                                order[1][0] = (graph_batchG[change_x + atom_offset][change_y + atom_offset]['mess_idx'], change_type)
                                
                            for x, y in hmol.change[1]:
                                if graph_batchG[x + atom_offset][y + atom_offset]['dir'] != 0: pdb.set_trace()
                                order[1][1].append( graph_batchG[x + atom_offset][y + atom_offset]['mess_idx'] )
                        else:
                            # break a ring
                            order[2] = break_rings[i]
                            for x, y, bond_type in hmol.order:
                                if hmol.mol_graph[x][y]['dir'] == 0:
                                    order[0].append( (graph_batchG[x + atom_offset][y + atom_offset]['mess_idx'], bond_type) )
                            #order[2] = [[], hmol.ring[1]]
                            #for i, (bond_i, bond_j) in enumerate(hmol.ring[0]):
                            #    tmp = []
                            #    for x, y in [bond_i, bond_j]:
                            #        tmp.append( graph_batchG[x + atom_offset][y + atom_offset]['mess_idx'] )
                            #    order[2][0].append(tmp)
                        
                            
                        # atom_change
                        for x in hmol.mol_graph.nodes:
                            if 'attach' in hmol.mol_graph.nodes[x]:
                                if 'charge_change' in hmol.mol_graph.nodes[x]:
                                    order[3].append( (x + atom_offset, hmol.mol_graph.nodes[x]['charge_change']) )
                                else:
                                    order[3].append( (x + atom_offset, 0) )
                       
                    all_orders.append(order)
        
        if not istest and not product:
            tree_batch = [x.mol_tree if x is not None else None for x in mol_batch]
            
            tree_scope = []
            last_num = 1
            for bid, tree in enumerate(tree_batch):
                offset = graph_scope[bid][0]
                if tree is None:
                    tree_scope.append( (last_num, 0) )
                    continue
                
                tree_scope.append( (last_num, len(tree.nodes)) )
                last_num += len(tree.nodes)
                
                for v in tree.nodes:           
                    cls = [x + offset for x in tree.nodes[v]['clq']]
                    egraph.append( cls )
                
            for i,hmol in enumerate(mol_batch):
                atom_offset = graph_scope[i][0]
                offset = tree_scope[i][0]
                if hmol is None:
                    all_orders.append([])
                else:
                    order = []
                    for x, y in hmol.order:
                        if y == -1:
                            order.append( (x + atom_offset, -1, -1) )
                        else:
                            order.append( (x + atom_offset, y + offset, vocab[hmol.mol_tree.nodes[y]['label']]) )
                    all_orders.append(order)
        
                
        egraph = create_pad_tensor(egraph)
        # Add atom mess index
        fgraph = torch.zeros(len(graph_batchG)+1, len(graph_batchG)+1).long()
            
        for u, v, attr in graph_batchG.edges(data=True):
            eid = attr['mess_idx']
            fgraph[u, v] = eid
      
        
        if use_brics and product:
            graph_tensors = graph_tensors[:4] + (fgraph, egraph, graph_scope)
            return (graph_batchG, tree_batchG),  (graph_tensors, tree_tensors), ggraph, all_orders
        else:
            graph_tensors = graph_tensors[:4] + (fgraph, egraph, graph_scope)
            return [graph_batchG],  [graph_tensors], None, all_orders
    
    
    @staticmethod
    def __append_mask(graphs, tensors, orders):
        if graphs is None: return graphs, tensors
        
        atom_mask = torch.ones(len(graphs)+1, 1).int()
        bond_mask = torch.ones(len(graphs.edges)+1, 1).int()
        
        dgraph = tensors[5].detach().numpy()
        
        for order in orders:
            
            for atom_idx, node_idx, _  in order:
                if node_idx == -1: continue
                
                try:        
                    tmp = dgraph[node_idx, :]
                    mask_atoms = torch.LongTensor( [a for a in tmp if a > 0 and ( 'attach' not in graphs.nodes[a] or graphs.nodes[a]['attach'] != 1 )] )
                except:
                    pdb.set_trace()
                
                atom_mask.scatter_(0, mask_atoms.unsqueeze(1), 0)
        
        mask1 = torch.ones(len(graphs)+1, 1).int()
        mask2 = torch.zeros(len(graphs)+1, 1).int()
        masked_atoms = torch.where(atom_mask==0, atom_mask, mask2)
        masked_atoms = torch.where(atom_mask>0, masked_atoms, mask1)
        masked_atoms = masked_atoms.nonzero()[:,0]
        
        mess_list = []
        for a1 in masked_atoms[1:]:
            a1 = a1.item()
            mess = torch.LongTensor([graphs[a1][edge[1]]['mess_idx'] for edge in graphs.edges(a1)])
            mess_list.append(mess)
            
            mess = torch.LongTensor([graphs[edge[1]][a1]['mess_idx'] for edge in graphs.edges(a1)])
            mess_list.append(mess)
        
        
        if len(mess_list) > 0:
            mess = torch.unique(torch.cat(mess_list, dim=0)).unsqueeze(1)
            try:
                bond_mask.scatter_(0, mess, 0)
            except:
                pdb.set_trace()
        
        tensors = tensors + (atom_mask, bond_mask)
        
        return graphs, tensors

    @staticmethod
    def tensorize_decoding(mol_batch, vocab, avocab, tree_buffer=None, extra_len=0, istest=False, use_atomic=False, use_feature=True):
        
        if not istest:
            tree_tensors, tree_batchG = MolTree.tensorize_graph([(x.mol_tree, x.mol) if x is not None else (None, None) for x in mol_batch], \
                                                                vocab,use_feature=use_feature)
            tree_scope = tree_tensors[-1]
            # Add anchor atom index
            cgraph = torch.zeros(len(tree_batchG.edges) + 1, 2).int()
            for u,v,attr in tree_batchG.edges(data=True):
                eid = attr['mess_idx']
                anchor = tree_batchG[u][v]['anchor']
                cgraph[eid, :len(anchor)] = torch.LongTensor(anchor)
                    
            # Add all atom index
            max_cls_size = vocab.max_len
            dgraph = torch.zeros(len(tree_batchG) + 1, max_cls_size).long()
            for v,attr in tree_batchG.nodes(data=True):
                bid = attr['batch_id']
                tree_batchG.nodes[v]['clq'] = cls = [x for x in attr['clq']]
                dgraph[v, :len(cls)] = torch.LongTensor(cls)
            tree_tensors = tree_tensors[:4] + (cgraph, dgraph, tree_scope)
        
        graph_tensors, graph_batchG = MolTree.tensorize_graph([(x.mol_graph, x.mol) if x is not None else (None, None) for x in mol_batch], \
                                                                  avocab, tree=False, tree_buffer=tree_buffer, use_atomic=use_atomic, use_feature=use_feature)
        
        graph_scope = graph_tensors[-1]
        
        # Add atom mess index
        egraph = torch.zeros(len(graph_tensors[0]), len(graph_tensors[0])).long()
        if graph_batchG is not None:
            for u, v, attr in graph_batchG.edges(data=True):
                eid = attr['mess_idx']
                egraph[u, v] = eid
        
        graph_tensors = graph_tensors[:4] + (egraph, None, graph_scope, None, None)
        
        return graph_tensors

    @staticmethod
    def tensorize_graph(graph_batch, vocab, tree=True, tree_buffer=None, atom_num=1, use_atomic=False, use_brics=False, extra_len=0, use_feature=False):
        if tree:
            fmess = [(0,0,0)]
            fnode = [0]
        elif not use_feature:
            fmess = [(0,0,0,0)]
            fnode = [0]
        else:
            fmess = [(0,0,0,0,0,0,0)]
            if not use_atomic:
                fnode = [(0,0,0,0,0,0)]
            else:
                fnode = [(0,0,0,0,0,0,0)]
        
        agraph,bgraph = [[]], [[]]
        scope = []
        edge_dict = {}
        all_G = []
        
        node_num = 1
        for bid, (G, mol) in enumerate(graph_batch):
            offset = len(fnode)
            if G is None:
                scope.append( (offset, 0) )
                continue
            else:
                scope.append( (offset, len(G)) )
            
            if tree_buffer is not None and tree_buffer[bid] is not None:
                fnode.extend( [fnode[0]] * len(G)  )
                agraph.extend( [[] for _ in range(len(G))] )
                continue
            else:
                fnode.extend( [None for v in G.nodes] )
            
            G = nx.convert_node_labels_to_integers(G, first_label=offset)
            all_G.append(G)
            
 
            for v, attr in G.nodes(data='label'):
                G.nodes[v]['batch_id'] = bid
                try:
                    if not use_feature and not tree:
                        fnode[v] = vocab[attr]
                    elif use_feature and not tree:
                        if not use_atomic:
                            fnode[v] = (vocab[attr], G.nodes[v]['valence'], G.nodes[v]['charge'], G.nodes[v]['num_h'], \
                                        int(G.nodes[v]['in_ring']), int(G.nodes[v]['aroma']))
                        else:
                            fnode[v] = (vocab[attr], G.nodes[v]['valence'], G.nodes[v]['charge'], G.nodes[v]['num_h'], \
                                        int(G.nodes[v]['in_ring']), int(G.nodes[v]['aroma']), G.nodes[v]['atomic_num'])
                            
                    else:
                        fnode[v] = fnode[0]
                    
                except Exception as e:
                    print(e)
                    fnode[v] = fnode[0]
                    
                if use_brics and not tree:
                    G.nodes[v]['brics_node_idx'] = G.nodes[v]['brics_node_idx'] + node_num
                
                agraph.append([])
            
            for u, v, attr in G.edges(data='label'):
                if tree:
                    fmess.append( (u, v, 0) )
                elif use_feature:
                    fmess.append( (u, v, attr, G[u][v]['is_conju'], G[u][v]['in_ring'], G[u][v]['is_aroma'], G[u][v]['dir']) )
                else:
                    fmess.append( (u, v, attr, G[u][v]['dir']) )
                
                edge_dict[(u, v)] = eid = len(edge_dict) + 1
                G[u][v]['mess_idx'] = eid
                
                if tree:
                    anchor = G[u][v]['anchor']
                    G[u][v]['anchor'] = [x+atom_num for x in anchor]
                
                agraph[v].append(eid)
                bgraph.append([])
                
            for u, v in G.edges():
                eid = edge_dict[(u, v)]
                for w in G.predecessors(u):
                    if w == v: continue
                    bgraph[eid].append( edge_dict[(w, u)] )
                #if 25 in bgraph[eid]: pdb.set_trace()
            if tree:
                atom_num += max([max(G.nodes[idx]['clq']) for idx in G.nodes]) + 1
            elif use_brics:
                node_num += len(set(G.nodes[u]['brics_node_idx'] for u in G.nodes))
                
        try:
            fnode = torch.LongTensor(fnode)
        except:
            fnode[0] = [0] * len(fnode[1])
            fnode = torch.LongTensor(fnode)
            
        fmess = torch.LongTensor(fmess)
        agraph = create_pad_tensor(agraph, extra_len=extra_len)
        bgraph = create_pad_tensor(bgraph, extra_len=extra_len)
        
        if len(all_G) > 0:
            return (fnode, fmess, agraph, bgraph, scope), nx.union_all(all_G)
        else:
            return (fnode, fmess, agraph, bgraph, scope), None
        
def identify_react_ids(product_mol, react_mols):
    product_bonds_with_atommap, _ = get_bonds_atommap(product_mol)
    
    reacts_atom_with_atommap = {}
    for i, mol in enumerate(react_mols):
        reacts_atom_with_atommap[i] = get_idx_from_mapnum(mol)
    
    react_id_edges = {}
    for product_bond_num, val in product_bonds_with_atommap.items():
        product_bond_idx = val[:2]
        product_bond_type = val[2]

        for i in range(len(react_mols)):
            if product_bond_num[0] in reacts_atom_with_atommap[i] or product_bond_num[1] in reacts_atom_with_atommap[i]:
                if product_bond_idx not in react_id_edges:
                    react_id_edges[product_bond_idx] = []

                react_id_edges[product_bond_idx].append(i)
    
    return react_id_edges

#def identify_revise_edges(product_mol, react_mol):
#    product_bonds_with_atommap, _ = get_bonds_atommap(product_mol)
#    react_bonds_with_atommap, react_bonds_without_atommap = get_bonds_atommap(react_mol)
#    react_atom_with_atommap = get_idx_from_mapnum(react_mol)
#    
#    charge_change_atoms = []
#    react_connect_atoms = set()
#    product_connect_atoms = set()
#    
#    for atom in product_mol.GetAtoms():
#        mapnum = atom.GetAtomMapNum()
#        atom_idx = atom.GetIdx()
#        charge = atom.GetFormalCharge()
#        
#        react_atom_id = react_atom_with_atommap[mapnum]
#        react_charge  = react_mol.GetAtomWithIdx(react_atom_id).GetFormalCharge()
#        
#        if react_charge != charge:
#            #pdb.set_trace()
#            react_connect_atoms.add( react_atom_id )
#            charge_change_atoms.append( (atom_idx, charge, react_charge) )
#            product_connect_atoms.add( atom_idx )
#        
#        for bond in react_mol.GetAtomWithIdx(react_atom_id).GetBonds():
#            if bond.GetEndAtom().GetAtomMapNum() == 0 or bond.GetBeginAtom().GetAtomMapNum() == 0:
#                react_connect_atoms.add( react_atom_id )
#                product_connect_atoms.add( atom_idx )
#                continue
#        
#    # get the bond change
#    delete_change_edges = []
#    add_edges = []
#    for product_bond_num, val in product_bonds_with_atommap.items():
#        product_bond_idx = val[:2]
#        product_bond_type = val[2]
#                        
#        add = False
#        if product_bond_num not in react_bonds_with_atommap:
#            delete_change_edges.append((product_bond_idx, 0))
#            add = True    
#        else:
#            react_val = react_bonds_with_atommap[product_bond_num]
#            react_bond_type = react_val[2]
#            
#            if react_bond_type[0] != product_bond_type[0] and not (react_bond_type[1] == product_bond_type[1] == 1):
#                delete_change_edges.append((product_bond_idx, react_bond_type[0]))
#                add = True
#
#        if add:
#            product_connect_atoms.add( product_bond_idx[0] )
#            product_connect_atoms.add( product_bond_idx[1] )
#
#            for atom_num in product_bond_num:
#                if atom_num in react_atom_with_atommap:
#                    react_connect_atoms.add(react_atom_with_atommap[atom_num])
#    
#    for bond, val in react_bonds_without_atommap.items():
#        bond_idx = bond[1:]
#        
#        for aidx in bond_idx:
#            atom = react_mol.GetAtomWithIdx(aidx)
#            atom_mapnum = atom.GetAtomMapNum()
#            if atom_mapnum > 0: react_connect_atoms.add(aidx)
#            
#        add_edges.append((bond_idx, None))
#    
#    react_connect_atoms = sorted(list(react_connect_atoms))
#    product_connect_atoms = sorted(list(product_connect_atoms))
#    
#    return delete_change_edges, add_edges, charge_change_atoms, react_connect_atoms, product_connect_atoms
#
#
#def update_revise_atoms(product_tree, react_tree):
#    product_mol = product_tree.mol
#    
#    react_mol = react_tree.mol
#    
#    delete_change_edges, add_edges, charge_change_atoms, react_connect_atoms, product_connect_atoms = identify_revise_edges(product_mol, react_mol)
#    
#    #if len(product_connect_atoms) == 0: pdb.set_trace()
#    product_tree.set_revise(delete_change_edges, product=True, charge_change_atoms=charge_change_atoms, connect_atoms=product_connect_atoms)
#    #pdb.set_trace()
#    #product_tree.set_react_ids(react_id_edges)
#    
#    react_tree.set_revise(add_edges, product=False, connect_atoms=react_connect_atoms)
#
#
##def update_revise_atoms(product_tree, react_trees):
##    product_mol = product_tree.mol
##    
##    react_mols = [react_tree.mol for react_tree in react_trees]
##    
##    react_id_edges, delete_change_edges, add_edges, charge_change_atoms, react_connect_atoms, product_connect_atoms = identify_revise_edges(product_mol, react_mols)
##    
##    #if len(product_connect_atoms) == 0: pdb.set_trace()
##    product_tree.set_revise(delete_change_edges, product=True, charge_change_atoms=charge_change_atoms, connect_atoms=product_connect_atoms)
##    #pdb.set_trace()
##    product_tree.set_react_ids(react_id_edges)
##    
##    for i, react_tree in enumerate(react_trees):
##        react_tree.set_revise(add_edges[i], product=False, connect_atoms=react_connect_atoms[i])
#    
#    
#def get_synthon_tree(react_tree):
#    #synthon_trees = []
#    
#    #for tree in react_trees:
#    mol = graph_to_mol(react_tree.mol_graph, keep_atom=False)
#    synthon_tree = MolTree(Chem.MolToSmiles(mol))
#    #synthon_trees.append(MolTree(Chem.MolToSmiles(mol)))
#    return synthon_tree
#    

def identify_revise_edges(product_mol, react_mol, debug=False):
    product_bonds_with_atommap, _ = get_bonds_atommap(product_mol)
    react_bonds_with_atommap, react_bonds_without_atommap = get_bonds_atommap(react_mol)
    react_atom_with_atommap = get_idx_from_mapnum(react_mol)
    
    charge_change_atoms = []
    react_connect_atoms = set()
    product_connect_atoms = set()
    
    for atom in product_mol.GetAtoms():
        mapnum = atom.GetAtomMapNum()
        atom_idx = atom.GetIdx()
        charge = atom.GetFormalCharge()
        
        react_atom_id = react_atom_with_atommap[mapnum]
        react_charge  = react_mol.GetAtomWithIdx(react_atom_id).GetFormalCharge()
        
        if react_charge != charge:
            #pdb.set_trace()
            react_connect_atoms.add( react_atom_id )
            charge_change_atoms.append( (atom_idx, charge, react_charge) )
            product_connect_atoms.add( atom_idx )
        
        for bond in react_mol.GetAtomWithIdx(react_atom_id).GetBonds():
            if bond.GetEndAtom().GetAtomMapNum() == 0 or bond.GetBeginAtom().GetAtomMapNum() == 0:
                react_connect_atoms.add( react_atom_id )
                product_connect_atoms.add( atom_idx )
                continue
        
    # get the bond change
    delete_change_edges = []
    add_edges = []
    if debug:
        print(product_bonds_with_atommap)
        print(react_bonds_with_atommap)
        
    for product_bond_num, val in product_bonds_with_atommap.items():
        product_bond_idx = val[:2]
        product_bond_type = val[2]
        
        #if product_bond_num[0] in react_atom_with_atommap or product_bond_num[1] in react_atom_with_atommap:
        #    if product_bond_idx not in react_id_edges:
        #        react_id_edges[product_bond_idx] = []
        #        
        #    react_id_edges[product_bond_idx].append(i)
                        
        add = False
        if product_bond_num not in react_bonds_with_atommap:
            delete_change_edges.append((product_bond_idx, 0))
            add = True    
        else:
            react_val = react_bonds_with_atommap[product_bond_num]
            react_bond_type = react_val[2]
            
            if react_bond_type[0] != product_bond_type[0] and not (react_bond_type[1] == product_bond_type[1] == 1):
                delete_change_edges.append((product_bond_idx, react_bond_type[0]))
                add = True

        if add:
            product_connect_atoms.add( product_bond_idx[0] )
            product_connect_atoms.add( product_bond_idx[1] )

            for atom_num in product_bond_num:
                if atom_num in react_atom_with_atommap:
                    react_connect_atoms.add(react_atom_with_atommap[atom_num])
    
    for bond, val in react_bonds_without_atommap.items():
        bond_idx = bond[1:]
        
        for aidx in bond_idx:
            atom = react_mol.GetAtomWithIdx(aidx)
            atom_mapnum = atom.GetAtomMapNum()
            if atom_mapnum > 0: react_connect_atoms.add(aidx)
            
        add_edges.append((bond_idx, None))
    
    react_connect_atoms = sorted(list(react_connect_atoms))
    product_connect_atoms = sorted(list(product_connect_atoms))
    
    return delete_change_edges, add_edges, charge_change_atoms, react_connect_atoms, product_connect_atoms
    

def update_revise_atoms(product_tree, react_tree, shuffle=False, use_dfs=True):
    product_mol = product_tree.mol
    
    react_mol = react_tree.mol
    
    delete_change_edges, add_edges, charge_change_atoms, react_connect_atoms, product_connect_atoms = identify_revise_edges(product_mol, react_mol)
    
    if len(product_connect_atoms) == 0: raise ValueError("cannot have zero connect atoms")
    product_tree.set_revise(delete_change_edges, product=True, charge_change_atoms=charge_change_atoms, connect_atoms=product_connect_atoms)
    
    #product_tree.set_react_ids(react_id_edges)
    react_tree.set_revise(add_edges, product=False, connect_atoms=react_connect_atoms, shuffle=shuffle, use_dfs=use_dfs)

def get_synthon_trees(react_tree):
    mol = graph_to_mol(react_tree.mol_graph, keep_atom=False)
    synthon_tree = MolTree(Chem.MolToSmiles(mol))
    return synthon_tree

def get_comb_bonds(bonds, label=None):
    combs = []
    gt_label = -1
    for i, bond_i in enumerate(bonds):
        for j, bond_j in enumerate(bonds[i+1:]):
            combs.append( (bond_i, bond_j) )
            if label == (i, j):
                gt_label = len(combs) - 1
    return combs, gt_label


if __name__ == "__main__":
    #import sys
    #lg = rdkit.RDLogger.logger() 
    #lg.setLevel(rdkit.RDLogger.CRITICAL)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--train", type=str, default="./data/logp06/", help="the path of training data")
    parser.add_argument("-o", "--out", type=str, default="./data/vocab.txt", help="the path of vocabulary")
    args = parser.parse_args()
    
    cset = set()
    for file_name in os.listdir(args.train):
        if not file_name.endswith("pkl"): continue
        with open(os.path.join(args.train, file_name), 'rb') as f:
            pairs_data = pickle.load(f)
            for molx, moly, path in pairs_data:
                cset.update(set([label for _, label in molx.mol_tree.nodes(data='label')]))
                cset.update(set([label for _, label in moly.mol_tree.nodes(data='label')]))
     
    with open(args.out, 'w') as f:
        for word in cset:
            f.write("%s\n" % word)
