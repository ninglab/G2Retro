import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import sys
import math
import time
import torch
import pickle
import torch.nn as nn
import argparse
import rdkit
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from molsynthon import MolSynthon
from vocab import Vocab, common_atom_vocab
from datautils import PairTreeFolder, MolTreeFolder
from torch.nn import DataParallel
from chemutils import is_sim
import random
import pdb

device = "cuda:0" if torch.cuda.is_available() else "cpu"
path = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":

    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default="../data/dfs_tensors_with_class.pkl", help='data path to training data')
    parser.add_argument('--vocab', type=str, default="../data/vocab.txt", help='data path to substructure vocabulary')
    parser.add_argument('--valid', type=str, default="../data/valid.csv", help='data path to substructure vocabulary')
    parser.add_argument('--save_dir', type=str, default=path+"/result/", help='data path to the directory used to save trained models')
    parser.add_argument('--load_epoch', type=int, default=0, help='an interger used to control the loaded model (i.e., if load_epoch==1000, '+\
                        'the model save_dir+1000.pkl would be loaded)')
    parser.add_argument('--ncpu', type=int, default=10, help='the number of cpus')
    
    # size of model
    parser.add_argument('--size', type=int, default=70000, help='size of training data')
    parser.add_argument('--hidden_size', type=int, default=32, help='the dimension of hidden layers')
    parser.add_argument('--batch_size', type=int, default=256, help='the number of molecule pairs in each batch')
    parser.add_argument('--latent_size', type=int, default=32, help='the dimention of latent embeddings')
    parser.add_argument('--embed_size', type=int, default=32, help='the dimention of substructure embedding')
    parser.add_argument('--depthG', type=int, default=5, help='the depth of message passing in graph encoder')
    parser.add_argument('--depthT', type=int, default=3, help='the depth of message passing in tree encoder')
  
    parser.add_argument('--use_edit', action='store_true')
    parser.add_argument('--use_brics', action='store_true')
    parser.add_argument('--sum_pool', action='store_false')  
    parser.add_argument('--use_feature', action='store_false')
    parser.add_argument('--use_attention', action='store_true')
    parser.add_argument('--update_embed', action='store_true')
    parser.add_argument('--shuffle', action='store_false')
    parser.add_argument('--use_product', action='store_false')
    parser.add_argument('--use_attachatom', action='store_true')
    parser.add_argument('--use_latent_attachatom', action='store_true')
    parser.add_argument('--use_class', action='store_true')
    parser.add_argument('--use_node_embed', action='store_true')
    parser.add_argument('--use_atomic', action='store_false')
    parser.add_argument('--reduce_dim', action='store_true')
    
    parser.add_argument('--network_type', type=str, default='gcn')
    parser.add_argument('--add_ds', action='store_true', help='a boolean used to control whether adding the embedding of disconnection site '+\
                        'into the latent embedding or not.')
    parser.add_argument('--clip_norm', type=float, default=50.0, help='')
    parser.add_argument('--use_tree', action='store_true')
    
    # control the learning process
    parser.add_argument('--total_step', type=int, default=-1, help='the number of epochs')
    parser.add_argument('--epoch', type=int, default=100, help='the number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
    parser.add_argument('--print_iter', type=int, default=20)
    parser.add_argument('--save_iter', type=int, default=3000)
    
    args = parser.parse_args()
    # make directory
    max_epoch = 0 
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    else:
        files = os.listdir(args.save_dir)
        for f in files:
            if "optim" not in f:
                epoch = int(f.split("-")[1])
                if epoch > max_epoch: max_epoch = epoch
    
    # read vocabulary
    try:
        vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
        vocab = Vocab(vocab)
    except:
        if os.path.exists(os.path.dirname(args.vocab)):
            vocab = args.vocab
        else:
            raise ValueError("directory of path for vocabulary does not exist: %s" % args.vocab)
    avocab = common_atom_vocab
    
    valid_data = []
    with open(args.valid) as f:
        for line in f.readlines()[1:]:
            s = line.strip("\r\n ").split(",")
            smiles = s[2].split(">>")
            valid_data.append((int(s[0]), s[1], smiles[1], smiles[0]))
    random.seed(10)
    random.shuffle(valid_data)
    valid_loader = MolTreeFolder(valid_data[:2000], vocab, avocab, num_workers=args.ncpu, use_class=args.use_class, del_center=False, batch_size=args.batch_size, usepair=True, use_atomic=args.use_atomic)
    valid_loader.shuffle = False
    valid_batches = [tmp for tmp in valid_loader]
    # load data loader
    
    args.usepair = False
    args.del_center = True
    t1 = time.time()
    # build the model
    model = MolSynthon(vocab, avocab, args)
    print(model)
    
    loader = PairTreeFolder(args.train, vocab, avocab, args, is_train_center=False)

    # load previous trained model
    load_epoch = args.load_epoch if args.load_epoch > 0 else max_epoch 
    if os.path.exists(args.save_dir + "/model_synthon.iter-" + str(load_epoch)):
        model.load_state_dict(torch.load(args.save_dir + "/model_synthon.iter-" + str(load_epoch), map_location=device))
    else:
        load_epoch = 0
    
    print("Model #Params: {0}K".format(sum([x.nelement() for x in model.parameters()]) / 1000))
    
    # if we load a trained model
    lr = args.lr
    if load_epoch == 0: start_epoch = 0
    else:
        loader.epoch = load_epoch + 1
        start_epoch = load_epoch + 1
        
    total_step = args.size // args.batch_size * start_epoch
    if load_epoch > 0: lr = lr * (args.anneal_rate ** (total_step // args.anneal_iter))
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.9, threshold=0.01, verbose=True)
    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))
    
    losses = np.zeros(3)
    acc_rec = np.zeros(3)
    nums = np.zeros(3)
    t1 = time.time()
    
    last_ie = -1
    optim_val_acc, optim_epoch = 0, 0
    for it, (batch, ie) in enumerate(loader):
        
        with torch.autograd.set_detect_anomaly(True):
            total_step += 1
            model.zero_grad()
            
            try:
                total_loss, loss, acc, rec, num = model(*batch, total_step)
                total_loss.backward()
            except Exception as e:
                print(e)
                pdb.set_trace()
                continue
                
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
        
        losses = losses + np.array([float(total_loss)]+[float(l) for l in loss])
        nums += np.array(num)
        acc_rec = acc_rec + np.array(acc+rec)
        
        # print loss and accuracy
        if total_step % args.print_iter == 0:
            t2 = time.time()
            losses /= args.print_iter
            acc_rec /= args.print_iter
            nums /= args.print_iter
             
            s = "[%d/%d/%d] timecost: %.2f, Loss: %.3f, " % (total_step, ie, it, t2-t1, losses[0])
            s = s + "topo(%d, %d): (%.4f, %.4f, %.4f), " % (nums[1], nums[2], losses[2], acc_rec[1], acc_rec[2])
            s = s + "node(%d): (%.4f, %.4f), " % (nums[0], losses[1], acc_rec[0])
            s = s + "PNorm: %.2f, GNorm: %.2f" % (param_norm(model), grad_norm(model))
            print(s)
            t1 = t2
            sys.stdout.flush()
            losses *= 0
            nums *= 0
            acc_rec *= 0
        
        # save model
        if ie % 5 == 0 and ie != 0:
            torch.save(model.state_dict(), args.save_dir + "/model_synthon.iter-" + str(ie))
        
        if last_ie != ie and ie >= 20:
            val_accs, val_count = 0, 0
            for valid_batch in valid_batches:
                react_smiles = valid_batch[5]
                smiles = model.test_synthon(*valid_batch[:5], tmp=valid_batch[5:])
                tidx = -1
                for idx, smile in enumerate(react_smiles):
                    if idx in valid_batch[-1]: continue
                    tidx += 1
                    if len(smiles[tidx]) == 0: continue
                    elif is_sim(smiles[tidx][0], smile): val_accs += 1
                val_count += len(react_smiles)
                
            top_1_acc = val_accs / val_count
            print("[valid/%d]: top1: %.4f" % (ie, top_1_acc))
            scheduler.step(top_1_acc)
            
            last_ie = ie
            if top_1_acc - optim_val_acc >= 0.002 and ie >= 20:
                optim_val_acc = top_1_acc
                optim_epoch = ie
                print("save optimal model at epoch %d" % (ie))
                torch.save(model.state_dict(), args.save_dir + "/model_synthon_optim.pt")
                
    torch.save(model.state_dict(), args.save_dir + "/model_synthon.iter-" + str(args.epoch))
    print("final optimal validation accuracy: %.4f at epoch %d" % (optim_val_acc, optim_epoch))
