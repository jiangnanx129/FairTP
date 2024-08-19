import os
import argparse
import numpy as np
import pandas as pd
import pickle
import json
import random

def adj_initial_sample(adj_path, sample_list):
    out_dir1 = './data/sd/'
    # adj采样，一直不变，从到到尾都是那450个节点
    raw_adj = np.load(adj_path)

    raw_adj_all1 = raw_adj
    raw_adj_all1[raw_adj_all1 != 0] = 1
    np.save(os.path.join(out_dir1, 'sd_rn_adj_all1'), raw_adj_all1)

    new_adj = raw_adj[sample_list] # 8k*8k, select 716*8k
    new_adj = new_adj[:,sample_list]
    print("新矩阵：", new_adj.shape) # 返回 716*716 的邻接矩阵
    
    np.save(os.path.join(out_dir1, 'adj_initial200'), new_adj) # 716
    new_adj[new_adj != 0] = 1
    np.save(os.path.join(out_dir1, 'sd_rn_adj_all1'), new_adj)
    print("finish!")




if __name__ == '__main__':
    adj_initial_sample('./data/sd/sd_rn_adj.npy')
