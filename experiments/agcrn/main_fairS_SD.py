import os
import argparse
import numpy as np
import json
import pickle

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)
'''
不是固定不变，固定的节点一直在换采样方式，
1. 每T次换采样
2. 每个batch换采样
'''
from src.models.agcrnS import AGCRN
from src.engines.agcrn_engineS_SD import AGCRN_Engine
from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset, get_dataset_info, load_dataset2
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

# record_s2018_HK_3_1e-2_S
def get_config():
    parser = get_public_config()
    parser.add_argument('--embed_dim', type=int, default=10)
    parser.add_argument('--rnn_unit', type=int, default=64) # y 64
    parser.add_argument('--num_layer', type=int, default=2) # y 2
    parser.add_argument('--cheb_k', type=int, default=2) # y 2

    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=0)
    parser.add_argument('--clip_grad_value', type=float, default=0)
    parser.add_argument('--yl_values', type=float, default=0.05) # 优劣标签
    # add
    parser.add_argument('--T_dynamic', type=int, default=3) # 动态时间为3个epoch
    parser.add_argument('--sam_num', type=int, default='200', help='sample sum')
    
    args = parser.parse_args()

    log_dir = './experiments/{}/{}/'.format(args.model_name, args.dataset) # SD/HK
    logger = get_logger(log_dir, __name__, 'record_s{}_SD_3_1e-3_S.log'.format(args.seed))
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

    data_path, _, node_num = get_dataset_info(args.dataset)
    
    # 初始化的分布，初始采样，baseline不变，Fairness每个batch改变！
    # ptr1 = np.load(os.path.join(data_path, args.years, 'his_initial450.npz'), allow_pickle=True) # his.npz为原始数据 (N,938,ci) ci=1
    ptr1 = np.load(os.path.join(data_path, args.years, 'his_initial200.npz'), allow_pickle=True) # his.npz为原始数据 (N,938,ci) ci=1
    sample_list, sample_dict, sample_map = ptr1['sample_list'], ptr1['sample_dict'].item(), ptr1['sample_map'].item() # 字典，加上item()
    with open('data/sd/sd_district2.json', 'rb') as file: # 打开 JSON 文件
        district13_road_index = pickle.load(file) # 读取文件内容, 字典
    # with open('data/hk/district13_roadindex.json', 'r') as file: # 打开 JSON 文件
    #     district13_road_index = json.load(file) # 读取文件内容, 字典


    district_nodes = get_district_nodes(sample_dict, sample_map) # 初始化的区域字典，键0-12，值list(0-449)

    # dataloader, scaler = load_dataset(data_path, args, logger) # 采样的数据
    dataloader, scaler = load_dataset2(data_path, args, logger) # 全数据

    model = AGCRN(node_num=args.sam_num,
                  input_dim=args.input_dim,
                  output_dim=args.output_dim,
                  embed_dim=args.embed_dim,
                  rnn_unit=args.rnn_unit,
                  num_layer=args.num_layer,
                  cheb_k=args.cheb_k
                  )
    
    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = None

    engine = AGCRN_Engine(T_dynamic = args.T_dynamic,
                          sample_num = args.sam_num,
                          district13_road_index = district13_road_index,
                          
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