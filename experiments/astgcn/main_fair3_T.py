import os
import argparse
import numpy as np
import json
import torch.nn as nn

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

from src.models.astgcn_fair3 import ASTGCN
from src.engines.astgcn_450.astgcn_engine_fair3_no5 import ASTGCN_Engine
from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info, load_dataset2 # 2是complete数据
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
    parser.add_argument('--order', type=int, default=3)
    parser.add_argument('--nb_block', type=int, default=2)
    parser.add_argument('--nb_chev_filter', type=int, default=64)
    parser.add_argument('--nb_time_filter', type=int, default=64)
    parser.add_argument('--time_stride', type=int, default=1)

    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=1e-4)
    parser.add_argument('--clip_grad_value', type=float, default=5)
    


    parser.add_argument('--T_dynamic', type=int, default=10) # 动态时间为3个epoch
    parser.add_argument('--sam_num', type=int, default='450', help='sample sum')
    parser.add_argument('--yl_values', type=float, default=0.0354) # 优劣标签, 优劣阈值0.3
    parser.add_argument('--d_lrate', type=int, default=1e-4, help='dis lr') # 鉴别器lr
    parser.add_argument('--n_number', type=int, default='300', help='counts of sample') # C938_nnumber
    parser.add_argument('--a_loss3', type=float, default= 0.01)
    parser.add_argument('--a_loss4', type=float, default= 5)
    parser.add_argument('--type', type=str, default="greedy2 " )

    args = parser.parse_args()

    log_dir = './experiments/{}/{}/'.format(args.model_name, args.dataset)
    logger = get_logger(log_dir, __name__, 'record_s{}_HK3_1.log'.format(args.seed))
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
    sam_num = args.sam_num
    
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(sam_num)
    
    adj = np.zeros((sam_num, sam_num), dtype=np.float32)
    for n in range(sam_num):
        idx = np.nonzero(adj_mx[n])[0]
        adj[n, idx] = 1

    L_tilde = normalize_adj_mx(adj, 'scalap')[0]
    cheb_poly = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in calculate_cheb_poly(L_tilde, args.order)] 
    

    # 初始化的分布，初始采样，baseline不变，Fairness每个batch改变！
    ptr1 = np.load(os.path.join(data_path, args.years, 'his_initial450.npz'), allow_pickle=True) # his.npz为原始数据 (N,938,ci) ci=1
    sample_list, sample_dict, sample_map = ptr1['sample_list'], ptr1['sample_dict'].item(), ptr1['sample_map'].item() # 字典，加上item()
    # with open('data/sd/sd_district.json', 'rb') as file: # 打开 JSON 文件
    #     sd_district = pickle.load(file) # 读取文件内容, 字典
    with open('data/hk/district13_roadindex.json', 'r') as file: # 打开 JSON 文件
        district13_road_index = json.load(file) # 读取文件内容, 字典

    

    district_nodes = get_district_nodes(sample_dict, sample_map) # 初始化的区域字典，键0-12，值list(0-449)


    # 读取全数据
    dataloader, scaler = load_dataset2(data_path, args, logger) # dataloader为938


    model = ASTGCN(node_num=args.sam_num,
                   input_dim=args.input_dim,
                   output_dim=args.output_dim,
                   device=args.device,
                   cheb_poly=cheb_poly,
                   order=args.order,
                   nb_block=args.nb_block,
                   nb_chev_filter=args.nb_chev_filter,
                   nb_time_filter=args.nb_time_filter,
                   time_stride=args.time_stride
                   )
    
    loss_fn = masked_mae
    criterion = nn.BCELoss()
    model_params = list(set(model.parameters()) - set(model.classif.parameters())) # list(model.BlockList.parameters() + model.final_conv.parameters())
    D_params = list(model.classif.parameters())
    
    optimizer = torch.optim.Adam(model_params, lr=args.lrate, weight_decay=args.wdecay)
    optimizer_D = torch.optim.Adam(D_params, lr = args.d_lrate, weight_decay = args.wdecay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=args.max_epochs, eta_min=1e-6)

    engine = ASTGCN_Engine(criterion = criterion,
                          d_lrate = args.d_lrate, # 鉴别器的lr
                          n_number = args.n_number, # 从n_number个组合中选出最合适的采样
                          model_params = model_params, # encoder+decoder的参数
                          D_params = D_params, # discriminator的参数
                          optimizer_D = optimizer_D, # 鉴别器的优化器
                          scheduler_D = scheduler_D, # 鉴别器的schedule
                          a_loss3 = args.a_loss3, # 鉴别器的优化器
                          a_loss4 = args.a_loss4, # 鉴别器的schedule
                          T_dynamic = args.T_dynamic,
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
        engine.train(args.T_dynamic, sample_list, sample_map, sample_dict, district_nodes, args.yl_values)
    else:
        total_nodes = sum(list(district13_road_index.values()), [])
        epoch=1
        engine.evaluate(args.mode, args.T_dynamic, sample_list, sample_map, sample_dict, district_nodes, args.yl_values, epoch, total_nodes)
        

if __name__ == "__main__":
    main()