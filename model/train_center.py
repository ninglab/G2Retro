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
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from molcenter import MolCenter
from vocab import Vocab, common_atom_vocab
from datautils import PairTreeFolder, MolTreeFolder
from torch.nn import DataParallel

device = "cuda:0" if torch.cuda.is_available() else "cpu"
path = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default="../data/dfs_tensors_with_class.pkl", help='data path to training data')
    parser.add_argument('--valid', type=str, default="../data/valid.csv", help='data path to validation data')
    
    parser.add_argument('--vocab', type=str, default="../data/vocab.txt", help='data path to substructure vocabulary')
    parser.add_argument('--save_dir', type=str, default="../result/", help='data path to the directory used to save trained models')
    parser.add_argument('--load_epoch', type=int, default=0, help='an interger used to control the loaded model (i.e., if load_epoch==1000, '+\
                        'the model save_dir+1000.pkl would be loaded)')
    parser.add_argument('--ncpu', type=int, default=8, help='the number of cpus')
    
    parser.add_argument('--size', type=int, default=70000, help='size of training data')
    parser.add_argument('--hidden_size', type=int, default=256, help='the dimension of hidden layers')
    parser.add_argument('--batch_size', type=int, default=256, help='the number of molecule pairs in each batch')
    parser.add_argument('--latent_size', type=int, default=32, help='the dimention of latent embeddings')
    parser.add_argument('--embed_size', type=int, default=32, help='the dimention of substructure embedding')
    parser.add_argument('--depthG', type=int, default=5, help='the depth of message passing in graph encoder')
    parser.add_argument('--depthT', type=int, default=3, help='the depth of message passing in tree encoder')
    
    parser.add_argument('--use_atomic', action="store_false", help='whether to use atomic number as feature (default value is True)')
    parser.add_argument('--use_node_embed', action="store_true", help='whether to use the substructure embedding in the prediction functions (default value is False)')
    parser.add_argument('--use_brics', action="store_true", help='whether to use brics substructures in the encoder (default value is False)')    
    parser.add_argument('--use_feature', action='store_false', help='whether to )
    parser.add_argument('--use_class', action='store_true', help='whether the reaction types are known')
    parser.add_argument('--update_embed', action='store_true')
    parser.add_argument('--shuffle', action='store_false')
    parser.add_argument('--use_product', action='store_false')
    parser.add_argument('--use_attachatom', action='store_true')
    parser.add_argument('--use_latent_attachatom', action='store_true')
    parser.add_argument('--sum_pool', action='store_false')
    parser.add_argument('--use_mess', action='store_true')
      
    parser.add_argument('--use_atom_product', action='store_true')
    
    parser.add_argument('--network_type', type=str, default='gcn')
    parser.add_argument('--clip_norm', type=float, default=10.0, help='')
    parser.add_argument('--use_tree', action='store_true')
    
    # control the learning process
    parser.add_argument('--epoch', type=int, default=150, help='the number of epochs')
    parser.add_argument('--total_step', type=int, default=-1, help='the number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
    parser.add_argument('--print_iter', type=int, default=20)
    parser.add_argument('--save_iter', type=int, default=3000)
    
    args = parser.parse_args()
    max_epoch = 0
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    else:
        files = os.listdir(args.save_dir)
        for f in files:
            if "optim" in f: continue
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
    
    # load data loader
    print("prepare dataloader....")
    loader = PairTreeFolder(args.train, vocab, avocab, args, is_train_center=True)
    print("finished")

    print("prepare validation set....")
    args.usepair = False
    args.del_center = True
    t1 = time.time()
    valid_data = []
    with open(args.valid) as f:
        for line in f.readlines()[1:]:
            s = line.strip("\r\n ").split(",")
            smiles = s[2].split(">>")
            valid_data.append((int(s[0]), s[1], smiles[1], smiles[0]))
    
    valid_loader = MolTreeFolder(valid_data, vocab, avocab, num_workers=args.ncpu, use_class=args.use_class, del_center=False, batch_size=len(valid_data), use_brics=args.use_brics, usepair=False, use_atomic=args.use_atomic)
    valid_loader.shuffle = False
    valid_loader.batch_size = 32
    valid_batches = [tmp[:3]+(tmp[-1],) for tmp in valid_loader]
    
    # build the model
    t2 = time.time()
    print("finish loading validation set: %.2f" % (t2-t1))
    model = MolCenter(vocab, avocab, args)
    print(model)
    
    # load previous trained model
    load_epoch = args.load_epoch if args.load_epoch > 0 else max_epoch
    if os.path.exists(args.save_dir + "/model_center.iter-" + str(load_epoch)):
        print("load %s/model_center.iter-%d" % (args.save_dir, load_epoch))
        model.load_state_dict(torch.load(args.save_dir + "/model_center.iter-" + str(load_epoch), map_location=device))
    else: load_epoch = 0
    
    # if we load a trained model
    lr = args.lr
    if load_epoch == 0: start_epoch = 0
    else:
        loader.epoch = args.epoch - load_epoch - 1
        start_epoch = load_epoch + 1
        
    total_step = args.size // args.batch_size * start_epoch
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.9, threshold=0.01, verbose=True)
    
    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))
    
    losses = np.zeros(4)
    acc_rec = np.zeros(5)
    nums = np.zeros(5)
    
    t1 = time.time()
    optim_epoch = 0
    optim_val_acc, last_ie = 0, 0
    for it, (batch, ie) in enumerate(loader):
        with torch.autograd.set_detect_anomaly(True):
            total_step += 1
            model.zero_grad()
            
            total_loss, loss, acc, rec, num = model(*batch)
            total_loss.backward()
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
             
            s = "[%d/%d/%d] timecost: %.2f, Loss: %.3f, " % (total_step, ie+load_epoch, it, t2-t1, losses[0])
            s = s + "center(%d): (%.4f, %.4f), bond_charge(%d,%d): (%.4f, %.4f, %.4f), " % (nums[0], losses[1], acc_rec[0], nums[1], nums[2], losses[2], acc_rec[1], acc_rec[3])
            s = s + "atom_charge(%d,%d): (%.4f, %.4f, %.4f), " % (nums[3], nums[4], losses[3], acc_rec[2], acc_rec[4])
            s = s + "PNorm: %.2f, GNorm: %.2f" % (param_norm(model), grad_norm(model))
            print(s)
            t1 = t2
            sys.stdout.flush()
            losses *= 0
            nums *= 0
            acc_rec *= 0

        # test the trained model on validation set
        if ie != last_ie:
            top_10_acc = []
            for valid_batch in valid_batches:
                tmp_top_10_acc, _, _ = model.validate_centers(*valid_batch)
                top_10_acc.append(tmp_top_10_acc)
            top_10_acc = np.concatenate(top_10_acc, axis=0)
            top_1_acc = np.sum(top_10_acc[:, 0] == 1) / top_10_acc.shape[0]
            top_3_acc = np.sum(top_10_acc[:, 2] == 1) / top_10_acc.shape[0]
            top_5_acc = np.sum(top_10_acc[:, 4] == 1) / top_10_acc.shape[0]
            
            scheduler.step(top_1_acc)
            print("[valid/%d] top1: %.4f, top3: %.4f, top5: %.4f" % (ie+load_epoch, top_1_acc, top_3_acc, top_5_acc))
            last_ie += 1
            
            # save the optimal model with the highest validation accuracy
            if top_1_acc - optim_val_acc >= 0.001 and ie + load_epoch >= 20:
                optim_val_acc = top_1_acc
                optim_epoch = ie + load_epoch
                print("save optimal model at epoch %d" % (ie + load_epoch))
                torch.save(model.state_dict(), args.save_dir + "/model_center_optim.pt")
                    
        # save model
        if ie % 5 == 0 and ie != 0:
            torch.save(model.state_dict(), args.save_dir + "/model_center.iter-" + str(ie+load_epoch))
        
    torch.save(model.state_dict(), args.save_dir + "/model_center.iter-" + str(args.epoch))
    print("final optimal validation accuracy: %.4f at epoch %d" % (optim_val_acc, optim_epoch))
