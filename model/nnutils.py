#!/usr/bin/env python3
# -*- coding: utf-8 
import pdb
import numpy as np
import torch
import torch.nn as nn
import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Variation(nn.Module):
    def __init__(self, input_size, z_size):
        super(Variation, self).__init__()
        self.input_size = input_size
        self.z_size=z_size   
        self.fc = nn.Sequential(
            nn.Linear(input_size, z_size),
            nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
            nn.Tanh(),
            nn.Linear(z_size, z_size),
            nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
            nn.Tanh(),
        )
        self.context_to_mu=nn.Linear(z_size, z_size) # activation???
        self.context_to_logsigma=nn.Linear(z_size, z_size) 
        
        self.fc.apply(self.init_weights)
        self.init_weights(self.context_to_mu)
        self.init_weights(self.context_to_logsigma)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):        
            m.weight.data.uniform_(-0.02, 0.02)
            m.bias.data.fill_(0)

    def forward(self, context):
        batch_size,_=context.size()
        context = self.fc(context)
        mu=self.context_to_mu(context)
        logsigma = self.context_to_logsigma(context) 
        std = torch.exp(0.5 * logsigma)
        
        epsilon = torch.randn([batch_size, self.z_size]).to(device)
        z = epsilon * std + mu  
        return z, mu, logsigma 


def unique_tensor(tensor):
    tensor = (tensor.data).cpu().numpy()
    unique_tensor = []
    visited = [-1 for _ in range(tensor.shape[0])]
    for i in range(tensor.shape[0] - 1):
        if visited[i] != -1: continue
        for j in range(i+1, tensor.shape[0]):
            if visited[j] != -1: continue
            boolean = np.allclose(tensor[i,:], tensor[j,:], atol=2e-07)
            if boolean:
                if visited[i] == -1:
                    unique_tensor.append(tensor[i,:])
                    visited[i] = len(unique_tensor) - 1
                
                visited[j] = len(unique_tensor) - 1
    
    for i in range(tensor.shape[0]):
        if visited[i] != -1: continue
        unique_tensor.append(tensor[i,:])
        visited[i] = len(unique_tensor) - 1

    unique_tensor = torch.tensor(np.stack(unique_tensor, axis=0)).to(device)
    return unique_tensor, visited
    
    
def create_pad_tensor(alist, extra_len=0):
    max_len = max([len(a) for a in alist]) + extra_len
    for a in alist:
        pad_len = max_len - len(a)
        a.extend([0] * pad_len)
    return torch.IntTensor(alist)
    
def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    
    try:
        target = source.index_select(dim, index.view(-1))
    except Exception as e:
        print(e)
        raise ValueError("cannot select index")
    return target.view(final_size)

def index_select_MUL(sources, dim, index):
    select_sources = []
    numpy_index = index.cpu().numpy()
    
    for source in sources:
        if type(source) is list:
            select_source = [source[i] for i in numpy_index]
        else:
            select_source = index_select_ND(source, dim, index)
        select_sources.append(select_source)
    return select_sources    
    
def MPL(x, cur_x_nei, cur_v_nei, W_g, U_g):
    new_v = torch.relu(W_g(torch.cat([cur_x_nei, cur_v_nei], dim=2)))
    v_nei = new_v.sum(dim=1)
    z = torch.relu(U_g(torch.cat([x, v_nei], dim=1)))
    
    return new_v, z
    
def bfs(node_stack, insert_stack, stop_stack, num):
    """ Breadth first search
    """
    temp = []
    for node, _, _ in node_stack[-num:]:
        if node is None: continue
        if len(stop_stack) == 0: node.fa_node = None
        num = len(temp)
        for i, neighbor in enumerate(node.keep_neighbors):
            if neighbor == node.fa_node: continue
            if neighbor.fa_node != node:
                neighbor.fa_node = node
            if i < len(node.keep_neighbors)-1: 
                temp.append((neighbor, True, -1))
            else:
                temp.append((neighbor, False, len(stop_stack)))
        
        if len(temp) == num:
            temp.append((None, False, len(stop_stack)))
        
        for n, neighbor in enumerate(node.insert_neighbors):
            if neighbor == node.fa_node: continue
            if neighbor.fa_node != node:
                neighbor.fa_node = node
            
            stop_stack.append((node, False, n, len(insert_stack)))
            
            if neighbor in node.subtrees:
                insert_stack.append((node, neighbor, node.subtrees[neighbor]))
            else:
                insert_stack.append((node, neighbor, None))
            
            temp.append((neighbor, False, len(stop_stack)))
        
        stop_stack.append((node, True, len(node.insert_neighbors), len(insert_stack)))
    
    num = len(temp)
    if num > 0:
        node_stack.extend(temp)
        bfs(node_stack, insert_stack, stop_stack, num)


## modified from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1

def _sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    
    batch_size = sequence_length.size(0)
    seq_range = torch.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    
    return seq_range_expand < seq_length_expand



def logsumexp(inputs, mask = None, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    if mask is not None: tmp = (inputs - s).exp() * mask.type(torch.FloatTensor).to(device)
    
    outputs = s + tmp.sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def variable_CE_loss(ori_logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch,) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """
    mask = _sequence_mask(sequence_length=length, max_len=ori_logits.size(1)).float().to(device)
    
    losses = - torch.gather(ori_logits, dim=1, index=target.unsqueeze(1)).view(-1) + logsumexp(ori_logits, dim=1, mask=mask)
    
    loss = losses.sum() / target.size(0)
    return loss

def variable_likelihood(logits, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch,) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """
    mask = _sequence_mask(sequence_length=length, max_len=logits.size(1)).float().to(device)
    
    logits = torch.exp(logits) * mask
    
    log_probs = torch.log( logits / torch.tile( logits.sum(1).unsqueeze(1), (1, logits.size(1))) + (1 - mask) + 1e-15)
    
    rank_log_probs = (log_probs - torch.min(log_probs) + 1e-15) * mask
    
    return rank_log_probs, log_probs

def get_likelihood(logits):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch,) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """
    if logits.shape[1] > 1:
        logits = torch.exp(logits)
        log_probs = torch.log( logits / torch.tile( logits.sum(1).unsqueeze(1), (1, logits.size(1))) + 1e-15)
        log_probs = torch.nan_to_num(log_probs)
    else:
        logits = torch.exp(-logits)
        log_probs_1 = torch.log( 1 / (1 + logits) )
        log_probs_0 = torch.log( 1 - 1 / (1 + logits) + 1e-15 )
        
        log_probs = torch.stack( [ log_probs_0, log_probs_1 ], dim=1)
    
    return log_probs



class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, depth):
        super(GCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.depth = depth
        
        self.W_g = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)) .to(device)
            
        self.output_mess = nn.Sequential(
            nn.Linear(depth * hidden_size, int(depth / 2) * hidden_size),
            nn.ReLU(),
            nn.Linear(int(depth / 2) * hidden_size, hidden_size)).to(device)
    
    def gcn(self, fmess, nei_message):
        g_input = torch.cat([fmess, nei_message], dim=1)
        messages = self.W_g( g_input )
        return messages
        
    def forward(self, fmess, mess_graph, mask=None):
        multi_layer_mess = []
        
        messages = torch.zeros(mess_graph.size(0), self.hidden_size).to(device)
        
        if mask is not None:
            fmess = index_select_ND(fmess, 0, mask.nonzero()[:, 0])
            mess_graph = index_select_ND(mess_graph, 0, mask.nonzero()[:, 0])
            
        for i in range(self.depth):
            if mess_graph.shape[1] > 0:
                try:
                    nei_message = index_select_ND(messages, 0, mess_graph)
                except:
                    pdb.set_trace()
                nei_message = nei_message.sum(dim=1)
            else:
                nei_message = torch.zeros((fmess.shape[0], self.hidden_size)).to(device)
                
            tmp_messages = self.gcn(fmess, nei_message)
            
            multi_layer_mess.append(tmp_messages)
            
            if mask is not None:
                messages[mask.nonzero()[:, 0], :] = tmp_messages
            else:
                messages = tmp_messages
            
            messages[0,:] = messages[0,:] * 0
        
        tmp_messages = torch.cat(multi_layer_mess, dim=1)
        tmp_messages = self.output_mess(tmp_messages)
        
        if mask is not None:
            messages = torch.zeros(mask.size(0), self.hidden_size).to(device)
            messages[mask.nonzero()[:, 0], :] = tmp_messages
        else:
            messages = tmp_messages
                
        messages[0,:] = messages[0,:] * 0 
        return messages
       


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, depth):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth

        self.W_z = nn.Linear(input_size + hidden_size, hidden_size).to(device)
        self.W_r = nn.Linear(input_size, hidden_size, bias=False).to(device)
        self.U_r = nn.Linear(hidden_size, hidden_size).to(device)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size).to(device)

    def get_init_state(self, fmess, init_state=None):
        h = torch.zeros(len(fmess), self.hidden_size, device=device)
        return h if init_state is None else torch.cat( (h, init_state), dim=0)

    def get_hidden_state(self, h):
        return h

    def GRU(self, x, h_nei):
        sum_h = h_nei.sum(dim=1)
        z_input = torch.cat([x,sum_h], dim=1)
        z = torch.sigmoid(self.W_z(z_input))

        r_1 = self.W_r(x).view(-1, 1, self.hidden_size)
        r_2 = self.U_r(h_nei)
        r = torch.sigmoid(r_1 + r_2)
        
        gated_h = r * h_nei
        sum_gated_h = gated_h.sum(dim=1)
        h_input = torch.cat([x,sum_gated_h], dim=1)
        pre_h = torch.tanh(self.W_h(h_input))
        new_h = (1.0 - z) * sum_h + z * pre_h
        return new_h

    def forward(self, fmess, bgraph, mask=None):
        h = torch.zeros(fmess.size(0), self.hidden_size, device=device)
        
        if mask is not None:
            fmess = index_select_ND(fmess, 0, mask.nonzero()[:, 0])
            bgraph = index_select_ND(bgraph, 0, mask.nonzero()[:, 0])
         
        for i in range(self.depth):
            h_nei = index_select_ND(h, 0, bgraph)
            tmp_h = self.GRU(fmess, h_nei)
            
            if mask is not None:
                h[mask.nonzero()[:, 0], :] = tmp_h
            else:
                h = tmp_h
            
            h[0,:] = h[0,:] * 0 

        return h

    def sparse_forward(self, h, fmess, submess, bgraph):
        mask = h.new_ones(h.size(0)).scatter_(0, submess, 0)
        h = h * mask.unsqueeze(1)
        for i in range(self.depth):
            h_nei = index_select_ND(h, 0, bgraph)
            sub_h = self.GRU(fmess, h_nei)
            h = index_scatter(sub_h, h, submess)
        return h
