import os
import argparse
import numpy as np
import pickle
import torch.nn as nn
import json

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

from src.models.fairgcn_fair1 import GCN_LSTM
from src.engines.fairgnn_engine_fair1 import FAIRGNN_Engine
from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset2, load_adj_from_numpy, get_dataset_info
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger
from src.utils.graph_algo import normalize_adj_mx

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()
    parser.add_argument('--n_filters', type=int, default=4) # hidden size 64
    parser.add_argument('--max_diffusion_step', type=int, default=2)
    parser.add_argument('--filter_type', type=str, default='doubletransition')
    parser.add_argument('--num_rnn_layers', type=int, default=2)
    parser.add_argument('--cl_decay_steps', type=int, default=2000)

    parser.add_argument('--lrate', type=float, default=1e-2)
    parser.add_argument('--wdecay', type=float, default=0)
    parser.add_argument('--clip_grad_value', type=float, default=5)
    # parser.add_argument('--yl_values', type=float, default=0.03) # 优劣标签
    parser.add_argument('--a_loss2', type=float, default=0.01)
    parser.add_argument('--a_loss3', type=float, default=0.1)
    
    args = parser.parse_args()

    log_dir = './experiments/{}/{}/'.format(args.model_name, args.dataset)
    logger = get_logger(log_dir, __name__, 'record_s{}_fairst_1.log'.format(args.seed)) # _nonomarlize
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

    data_path, adj_path, node_num = get_dataset_info(args.dataset) # 716
    logger.info('Adj path: ' + adj_path) 
    
    adj_mx = load_adj_from_numpy(adj_path) # 加载(350*350的矩阵)
    # if np.all(adj_mx == 0):
    #     print("aaaa")
    adj_mx = normalize_adj_mx(adj_mx, 'transition') # yes/no
    adj_mx = torch.from_numpy(adj_mx[0]).to(device)

    # # 初始化的分布，初始采样，baseline不变，Fairness每个batch改变！
    # ptr1 = np.load(os.path.join(data_path, args.years, 'his_initial450.npz'), allow_pickle=True) # his.npz为原始数据 (N,938,ci) ci=1
    # sample_list, sample_dict, sample_map = ptr1['sample_list'], ptr1['sample_dict'].item(), ptr1['sample_map'].item() # 字典，加上item()
    # with open('data/sd/sd_district2.json', 'rb') as file: # 打开 JSON 文件
    #     sd_district = pickle.load(file) # 读取文件内容, 字典
    
    # district_nodes = get_district_nodes(sample_dict, sample_map) # 初始化的区域字典，键0-12，值list(0-449)
    
    '''938 全数据分区情况'''
    with open('./data/hk/district13_roadindex.json', 'r') as file: # 打开 JSON 文件
        district13_road_index = json.load(file) # 读取文件内容, 字典，键0-12，值为list(0-937)


    dataloader, scaler = load_dataset2(data_path, args, logger) # 全数据

  
    model = GCN_LSTM(node_num=node_num,
                     input_dim=args.input_dim,
                     output_dim=args.output_dim,
                     device=device,
                     n_filters=args.n_filters,
                     adj = adj_mx
                  # adj_mx=adj_mx,
                  # max_diffusion_step=args.max_diffusion_step,
                  # filter_type=args.filter_type, # 邻接矩阵正则化方式
                  # num_rnn_layers=args.num_rnn_layers,
                  # cl_decay_steps=args.cl_decay_steps # 防止梯度爆炸
                  )

    # loss_fn = masked_mae
    # criterion = nn.BCEWithLogitsLoss() #nn.BCELoss() # 多分类+softmax   nn.CrossEntropyLoss() #  # 鉴别器损失
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
 
    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    steps = [10, 50, 90]  # CA: [5, 50, 90], others: [10, 50, 90]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=0.1, verbose=True)
    
    
    # G_params = list(model.GNN.parameters()) + list(model.classifier.parameters()) 
    # D_params = list(model.adv.parameters())
    # optimizer_G = torch.optim.Adam(G_params, lr = args.lrate, weight_decay = args.wdecay)
    # optimizer_D = torch.optim.Adam(D_params, lr = args.lrate, weight_decay = args.wdecay)
    # steps = [10, 50, 90]  # CA: [5, 50, 90], others: [10, 50, 90]
    # scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=steps, gamma=0.1, verbose=True)
    # scheduler_D = torch.optim.lr_scheduler.MultiStepLR(optimizer_D, milestones=steps, gamma=0.1, verbose=True)

    engine = FAIRGNN_Engine(device=device,
                          model=model,
                          dataloader=dataloader,
                          scaler=scaler,
                          sampler=None,
                          loss_fn=loss_fn,
                          lrate=args.lrate,
                          optimizer=optimizer,
                          scheduler=scheduler,

                        #   criterion=criterion, # 鉴别损失计算方式
                        #   optimizer=optimizer_G,
                        #   optimizer_D=optimizer_D,
                        #   scheduler=scheduler_G,
                        #   scheduler_D=scheduler_D,

                          clip_grad_value=args.clip_grad_value,
                          max_epochs=args.max_epochs,
                          patience=args.patience,
                          log_dir=log_dir,
                          logger=logger,
                          seed=args.seed,
                          a_loss2 = args.a_loss2, # 静态损失超参数
                          a_loss3 = args.a_loss3
                          )

    if args.mode == 'train':
        engine.train(district13_road_index)
    else:
        engine.evaluate(args.mode, district13_road_index, epoch=1)


if __name__ == "__main__":
    main()