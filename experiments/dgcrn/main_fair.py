import os
import argparse
import numpy as np
import pickle

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

from src.models.dgcrn import DGCRN
from src.engines.dgcrn_engine import DGCRN_Engine
from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from src.utils.graph_algo import normalize_adj_mx
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()
    parser.add_argument('--gcn_depth', type=int, default=2)
    parser.add_argument('--rnn_size', type=int, default=64)
    parser.add_argument('--hyperGNN_dim', type=int, default=16)
    parser.add_argument('--node_dim', type=int, default=40)
    parser.add_argument('--tanhalpha', type=int, default=3)
    parser.add_argument('--cl_decay_step', type=int, default=2000)
    parser.add_argument('--step_size', type=int, default=2500)
    parser.add_argument('--tpd', type=int, default=96)

    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--clip_grad_value', type=float, default=5)
    parser.add_argument('--yl_values', type=float, default=0.05) # 优劣标签
    parser.add_argument('--sam_num', type=int, default=200) # 
    args = parser.parse_args()

    log_dir = './experiments/{}/{}/'.format(args.model_name, args.dataset)
    logger = get_logger(log_dir, __name__, 'record_s{}_HK_1.log'.format(args.seed))
    logger.info(args)
    
    return args, log_dir, logger

def get_district_nodes(sample_dict, sample_map): # 2字典，sample_dict键(3,4)，值list(0-937); sample_map键0-449,值(0-938)
    district_nodes = {}
    for v, node_list in sample_dict.items():
        # select_list = []
        # for key, value in sample_map.items():
        #     if value in node_list:
        #         select_list.append(key)
        # print("selecy_list:",select_list)

        select_list  = [key for key, value in sample_map.items() if value in node_list]
        district_nodes[v] = select_list # 保存每个区域要取的节点（0-449），键为区域（0-12），值为list
    return district_nodes # 字典，键0-12，值list(0-449)


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)
    
    data_path, adj_path, node_num = get_dataset_info(args.dataset)

    logger.info('Adj path: ' + adj_path)

    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = normalize_adj_mx(adj_mx, 'doubletransition')
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    
    # 初始化的分布，初始采样，baseline不变，Fairness每个batch改变！
    ptr1 = np.load(os.path.join(data_path, args.years, 'his_initial200.npz'), allow_pickle=True) # his.npz为原始数据 (N,938,ci) ci=1
    sample_list, sample_dict, sample_map = ptr1['sample_list'], ptr1['sample_dict'].item(), ptr1['sample_map'].item() # 字典，加上item()
    with open('data/sd/sd_district.json', 'rb') as file: # 打开 JSON 文件
        sd_district = pickle.load(file) # 读取文件内容, 字典
    

    district_nodes = get_district_nodes(sample_dict, sample_map) # 初始化的区域字典，键0-12，值list(0-449)
    

    dataloader, scaler = load_dataset(data_path, args, logger)

    model = DGCRN(node_num=args.sam_num,
                  input_dim=args.input_dim,
                  output_dim=args.output_dim,
                  device=device,
                  predefined_adj=supports,
                  gcn_depth=args.gcn_depth,
                  rnn_size=args.rnn_size,
                  hyperGNN_dim=args.hyperGNN_dim,
                  node_dim=args.node_dim,
                  middle_dim=2,
                  list_weight=[0.05, 0.95, 0.95],
                  tpd=args.tpd,
                  tanhalpha=args.tanhalpha,
                  cl_decay_step=args.cl_decay_step,
                  dropout=args.dropout
                  )
    
    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = None

    engine = DGCRN_Engine(device=device,
                          model=model,
                          dataloader=dataloader,
                          scaler=scaler,
                          sampler=None,
                          loss_fn=loss_fn,
                          lrate=args.lrate,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          clip_grad_value=args.clip_grad_value,
                          max_epochs=args.max_epochs,
                          patience=args.patience,
                          log_dir=log_dir,
                          logger=logger,
                          seed=args.seed,
                          step_size=args.step_size,
                          horizon=args.horizon
                          )

    if args.mode == 'train':
        engine.train(args.yl_values, sample_map, district_nodes)
    else:
        engine.evaluate(args.mode, args.yl_values, sample_map, district_nodes, epoch=1)



if __name__ == "__main__":
    main()