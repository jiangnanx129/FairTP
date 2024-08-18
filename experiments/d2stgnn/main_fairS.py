import os
import argparse
import numpy as np
import json

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

from src.models.d2stgnn_fair3_T import D2STGNN # 涉及到采样adj_list, baseline每T次重新采样！
from src.engines.d2stgnn_200.d2stgnn_engineS import D2STGNN_Engine
from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset, load_dataset2, load_adj_from_numpy, get_dataset_info
from src.utils.graph_algo import normalize_adj_mx, calculate_cheb_poly
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
    parser.add_argument('--num_feat', type=int, default=1)
    parser.add_argument('--num_hidden', type=int, default=32)
    parser.add_argument('--node_hidden', type=int, default=12)
    parser.add_argument('--time_emb_dim', type=int, default=12)
    parser.add_argument('--layer', type=int, default=5)
    parser.add_argument('--k_t', type=int, default=3)
    parser.add_argument('--k_s', type=int, default=2)
    parser.add_argument('--gap', type=int, default=3) # 输出维度3，改成1
    parser.add_argument('--cl_epoch', type=int, default=3)
    parser.add_argument('--warm_epoch', type=int, default=30)
    parser.add_argument('--tpd', type=int, default=96)

    parser.add_argument('--lrate', type=float, default=2e-3)
    parser.add_argument('--wdecay', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--clip_grad_value', type=float, default=5)
    parser.add_argument('--yl_values', type=float, default=0.05) # 优劣标签

    # add
    parser.add_argument('--T_dynamic', type=int, default=3) # 动态时间为3个epoch
    parser.add_argument('--sam_num', type=int, default='200', help='sample sum')
   
    args = parser.parse_args()

    log_dir = './experiments/{}/{}/'.format(args.model_name, args.dataset)
    logger = get_logger(log_dir, __name__, 'record_s{}_HK_3_2e-3_S_b24.log'.format(args.seed))
    logger.info(args)
    logger.info("学习率")
    
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
    
    # 有adj要注意！！！
    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    adj_path = '/home/data/xjn/23largest_baseline/LargeST/data/hk/adj_938.npy' # 完整的邻接矩阵，不是采样过的
    logger.info('Adj path: ' + adj_path)
    
    adj_mx = load_adj_from_numpy(adj_path) # 读取邻接矩阵
    

    # 初始化的分布，初始采样，baseline不变，Fairness每个batch改变！
    ptr1 = np.load(os.path.join(data_path, args.years, 'his_initial200.npz'), allow_pickle=True) # his.npz为原始数据 (N,938,ci) ci=1
    
    sample_list, sample_dict, sample_map = ptr1['sample_list'], ptr1['sample_dict'].item(), ptr1['sample_map'].item() # 字典，加上item()
    # with open('data/sd/sd_district.json', 'rb') as file: # 打开 JSON 文件
    #     sd_district = pickle.load(file) # 读取文件内容, 字典
    with open('data/hk/district13_roadindex.json', 'r') as file: # 打开 JSON 文件
        district13_road_index = json.load(file) # 读取文件内容, 字典

    district_nodes = get_district_nodes(sample_dict, sample_map) # 初始化的区域字典，键0-12，值list(0-449)


    # dataloader, scaler = load_dataset(data_path, args, logger) # 注意在dataloader的函数中是200还是450
    dataloader, scaler = load_dataset2(data_path, args, logger) # 全数据

    cl_step = args.cl_epoch * dataloader['train_loader'].num_batch
    warm_step = args.warm_epoch * dataloader['train_loader'].num_batch

    model = D2STGNN(node_num=args.sam_num,
                    input_dim=args.input_dim,
                    output_dim=args.output_dim,
                    model_args=vars(args)
                    )
    
    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = None
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 38, 46, 54, 62, 70, 80], gamma=0.5) # 原始

    engine = D2STGNN_Engine(adj_mx = adj_mx, 
                           
                           sam_num = args.sam_num ,
                          T_dynamic = args.T_dynamic,
                          sample_num = args.sam_num,
                          district13_road_index = district13_road_index,

                          cl_step=cl_step,
                            warm_step=warm_step,
                            horizon=args.horizon,
                          
                          device=device,
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
                           seed=args.seed
                           )

    if args.mode == 'train':
        
        engine.train(sample_list, args.T_dynamic, args.yl_values, sample_map, district_nodes)
    else:
        engine.evaluate(args.mode, sample_list, args.T_dynamic, args.yl_values, sample_map, district_nodes, epoch=1)


if __name__ == "__main__":
    main()