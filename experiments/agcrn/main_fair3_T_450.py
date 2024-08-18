import os
import argparse
import numpy as np
import json
import torch.nn as nn

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

from src.models.agcrn_fairn3 import AGCRN
from src.engines.agcrn_450.agcrn_engine_fair3_T1_e2 import AGCRN_Engine # T1高效，T2三个list
from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset2, get_dataset_info
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
    parser.add_argument('--embed_dim', type=int, default=10)
    parser.add_argument('--rnn_unit', type=int, default=64) # y 64
    parser.add_argument('--num_layer', type=int, default=2) # y 2
    parser.add_argument('--cheb_k', type=int, default=2) # y 2

    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=0)
    parser.add_argument('--clip_grad_value', type=float, default=0)
    # parser.add_argument('--yl_values', type=float, default=0.05) # 优劣标签
    # add
    parser.add_argument('--T_dynamic', type=int, default=1) # 动态时间为3个epoch
    parser.add_argument('--sam_num', type=int, default='450', help='sample sum')
    parser.add_argument('--yl_values', type=float, default=0.0354) # 优劣标签, 优劣阈值0.3
    parser.add_argument('--d_lrate', type=int, default=1e-3, help='dis lr') # 鉴别器lr
    parser.add_argument('--n_number', type=int, default='300', help='counts of sample') # C938_nnumber
    parser.add_argument('--a_loss3', type=float, default= 0.01)
    parser.add_argument('--a_loss4', type=float, default= 5)
    parser.add_argument('--type', type=str, default="greedy2 " )
    parser.add_argument('--model_type', type=str, default="agcrn_fairn3_2 " )
    
    args = parser.parse_args()

    log_dir = './experiments/{}/{}/'.format(args.model_name, args.dataset) # SD/HK
    logger = get_logger(log_dir, __name__, 'record_s{}_HK3_disout7_10_e2.log'.format(args.seed))
    # _HK3_disout7_9_e3, T次一采样，贪心采样
    # _HK3_disout7_9_e2，每次12采样、，T次124采样，贪心采样。因为有不同的节点，但是某个节点可能累计有好几百，loss4_T越来越大
    # _HK3_disout7_9_e4，同e3，换采样方式，分布采样
    logger.info(args)
    logger.info("3 list low efficience")
    
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
    
    ptr1 = np.load(os.path.join(data_path, args.years, 'his_initial450.npz'), allow_pickle=True) # his.npz为原始数据 (N,938,ci) ci=1
    sample_list, sample_dict, sample_map = ptr1['sample_list'], ptr1['sample_dict'].item(), ptr1['sample_map'].item() # 字典，加上item()
    # with open('data/sd/sd_district.json', 'rb') as file: # 打开 JSON 文件
    #     sd_district = pickle.load(file) # 读取文件内容, 字典
    with open('data/hk/district13_roadindex.json', 'r') as file: # 打开 JSON 文件
        district13_road_index = json.load(file) # 读取文件内容, 字典

    district_nodes = get_district_nodes(sample_dict, sample_map) # 初始化的区域字典，键0-12，值list(0-449)

    # 读取全数据
    dataloader, scaler = load_dataset2(data_path, args, logger) # dataloader为938

    # dataloader, scaler = load_dataset(data_path, args, logger)

    model = AGCRN(node_num=args.sam_num,
                  input_dim=args.input_dim,
                  output_dim=args.output_dim,
                  embed_dim=args.embed_dim,
                  rnn_unit=args.rnn_unit,
                  num_layer=args.num_layer,
                  cheb_k=args.cheb_k
                  )
   
    
    loss_fn = masked_mae
    criterion = nn.BCELoss()

    # '''参数初始化'''
    # for p in model.parameters():
    #     if p.dim() > 1:
    #         torch.nn.init.xavier_uniform_(p)
    #     else:
    #         torch.nn.init.uniform_(p)


    model_params = list(set(model.parameters()) - set(model.classif.parameters()))

    print("1111111111111111")
   # model_params = list(model.node_embed.parameters()) + list(model.encoder.parameters()) + list(model.end_conv.parameters()) 
    #print("1111111111111111")
    D_params = list(model.classif.parameters())
    optimizer = torch.optim.Adam(model_params, lr=args.lrate, weight_decay=args.wdecay)
    optimizer_D = torch.optim.Adam(D_params, lr = args.d_lrate, weight_decay = args.wdecay)
    scheduler, scheduler_D = None, None

    # loss_fn = masked_mae
    # criterion = nn.BCELoss()
    # model_params = model.parameters()
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    # scheduler = None

    # D_params = model.classif.parameters()
    # optimizer_D = torch.optim.Adam(D_params, lr = args.d_lrate, weight_decay = args.wdecay)
    # scheduler_D = None

    engine = AGCRN_Engine(criterion = criterion,
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
        
        # engine.evaluate(args.mode,sample_list, sample_map, sample_dict, district_nodes, args.yl_values, epoch=1)


if __name__ == "__main__":
    main()