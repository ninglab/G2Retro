""" combine predictions from the models within the ensemble and evaluate them
"""
import argparse
import os
import numpy as np
from chemutils import is_sim, canonicalize
from multiprocessing import Pool
import math

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str)
parser.add_argument("--num", type=int)
parser.add_argument("--top_centers", type=str)
parser.add_argument("--top_synthons", type=str)
parser.add_argument("--mode", type=str)
parser.add_argument("--out", type=str)
parser.add_argument("--size", type=int, default=5008)
parser.add_argument("--batch", type=int, default=1002)
args = parser.parse_args()

sts = [i for i in range(0, args.size, args.batch)]
top_centers = [line.strip() for line in open(args.top_centers, 'r')]
top_synthons = [line.strip() for line in open(args.top_synthons, 'r')]

top_models = []
sqrt_num = int(math.sqrt(args.num))

for synthon_model in top_synthons:
    elems = synthon_model.split("_")
    hidden_sizeS = int(elems[0][6:])
    embed_sizeS = int(elems[1][5:])
    depthGS = int(elems[2][6:])
    
    for center_model in top_centers[:sqrt_num]:
        elems = center_model.split("_")
        hidden_sizeC = int(elems[0][6:])
        embed_sizeC = int(elems[2][5:])
        depthGC = int(elems[3][6:])
    
        top_model = "hidden" + "_".join([str(hidden_sizeC), str(hidden_sizeS)]) + "_nobrics_" + \
                    "embed" + "_".join([str(embed_sizeC), str(embed_sizeS)]) + "_" + \
                    "depthG" + "_".join([str(depthGC), str(depthGS)])
        top_models.append(top_model)
        if len(top_models) == args.num: break
    if len(top_models) == args.num: break
    
# ============== get all generated reactants ====================================
all_files = []

for i, st in enumerate(sts):
    all_files.append([])
    for f in os.listdir(args.dir):
        is_top = False
        for top_model in top_models:
            if top_model in f:
                is_top = True
        
        if is_top and "st"+str(st) in f and "pred" in f:
            all_files[-1].append(f)
    
    if len(all_files[-1]) < args.num:
        raise ValueError("%d is smaller than the required number %d for files starting from %d" % (len(all_files[-1]), args.num, st))
    all_files[-1] = all_files[-1][:args.num]

results = {}
data = {}

min_lh = 0

for st, files in zip(sts, all_files):
    for f in files:
        rank = 1
        last_idx = st-1
        fh = open(args.dir + f, 'r')
        raw_data = [line.strip().split(" ") for line in fh.readlines()]
        all_reacts = [tmp[4] for tmp in raw_data]
        fh.close()
        
        for i, tmp in enumerate(raw_data):
            idx, uspto_id, product, gt_react, react, lh = tmp
            
            idx = int(idx) + st
            lh = float(lh)
            if lh < min_lh: min_lh = lh
            if idx not in results:
                results[idx] = {}
                data[idx] = (uspto_id, product, gt_react)
                
            if idx != last_idx:
                rank = 1
                last_idx = idx
            
            try:
                if react in results[idx]:
                    results[idx][react].append((rank, lh))
                else:
                    results[idx][react] = [(rank, lh)]
            except:
                pdb.set_trace()
    
            rank += 1
        
        fh.close()

# ============== get top-10 reactants via re-ranking ============================
top_10_reacts = []
if args.mode == "rank":
    for i in range(len(results)):
        result = results[i]
        sum_rank_result = {}
        for react in result:
            sum_rank_result[react] = sum([1/tmp[0] for tmp in result[react]] + [1/20] * (len(files) - len(result[react])))
        top_10_react = sorted(list(sum_rank_result.keys()), key=lambda x: -1 * sum_rank_result[x])[:10]
        top_10_reacts.append(top_10_react)
elif args.mode == "lh":
    for i in range(len(results)):
        result = results[i]
        sum_lh_result = {}
        for react in result:
            sum_lh_result[react] = sum([tmp[1] for tmp in result[react]] + [min_lh] * (len(files) - len(result[react])))
        top_10_react = sorted(list(sum_lh_result.keys()), key=lambda x: -1 * sum_lh_result[x])[:10]
        
        top_10_reacts.append(top_10_react)

# ============ evaluate top-10 reactants ========================================
top_10_acc = np.zeros((len(top_10_reacts), 10))
acc_idxs = np.ones(len(top_10_reacts)) * 10
for i, top_10_react in enumerate(top_10_reacts):
    gt_react = data[i][2]
    
    for j, react in enumerate(top_10_react):
        if is_sim(gt_react, react):
            top_10_acc[i, j:] = 1
            acc_idxs[i] = j
            break
print("number of data: %d" % (len(results)))
avg_10_acc = np.mean(top_10_acc, axis=0)
for i in range(10):
    print("top-%d acc: %.4f" % (i, avg_10_acc[i]))

# =========== output top-10 reactants ===========================================
f = open(args.out, 'w')
for i, top_10_react in enumerate(top_10_reacts):
    tmp_data = data[i]
    for react in top_10_react:
        string = "%d " % (acc_idxs[i])
        string += " ".join(list(tmp_data)+[react])

        f.write(string + "\n")

f.close()
