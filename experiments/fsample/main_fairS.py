import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import json
import pickle
import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

from src.models.fsample_T import FSAMPLE # fsample4
from src.engines.fsample_200.fsample_engineS2 import FSAMPE_Engine # fsample_engine5
from src.utils.args import get_public_config # 引入公共参数
from src.utils.dataloader import load_dataset2, load_adj_from_numpy, get_dataset_info
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config() # 所有参数直接罗列！
    parser.add_argument('--n_filters', type=int, default=64) # fliter 32
    parser.add_argument('--max_diffusion_step', type=int, default=2)
    parser.add_argument('--filter_type', type=str, default='doubletransition')
    parser.add_argument('--num_rnn_layers', type=int, default=2)
    parser.add_argument('--cl_decay_steps', type=int, default=2000)
    parser.add_argument('--dis_output_dim', type=int, default=1) # 2

    parser.add_argument('--T_dynamic', type=int, default=3) # 动态时间为3个epoch
    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=0)
    parser.add_argument('--clip_grad_value', type=float, default=5) # 对模型的梯度进行裁剪，以确保它们的范数不超过指定的阈值
    parser.add_argument('--yl_values', type=float, default=0.05) # 优劣标签
   
    parser.add_argument('--sam_num', type=int, default='200', help='sample sum')
    args = parser.parse_args()

    log_dir = './experiments/{}/{}/'.format(args.model_name, args.dataset)
    logger = get_logger(log_dir, __name__, 'record_s{}_HK_3_1e-3_S2_b48_nos.log'.format(args.seed)) # 日志保存位置 5,0.05,0.1
    logger.info(args) # 日志添加参数信息
    logger.info("学习率None，验证不采样") 
    
    return args, log_dir, logger


def get_district_nodes(sample_dict, sample_map): # 2字典，sample_dict键0-12，值list(0-937); sample_map键0-449,值(0-938)
    district_nodes = {}
    for v, node_list in sample_dict.items():

        select_list  = [key for key, value in sample_map.items() if value in node_list]
        district_nodes[v] = select_list # 保存每个区域要取的节点（0-449），键为区域（0-12），值为list
    return district_nodes # 字典，键0-12，值list(0-449)


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed) #随机种子，确保实验结果复现
    device = torch.device(args.device) # 放到某个gpu上
    

    data_path, adj_path, node_num = get_dataset_info(args.dataset) # 数据集位置，邻接矩阵位置，采样节点总数
    # adj_path = '/home/data/xjn/23largest_baseline/LargeST/data/sd/sd_rn_adj.npy' # 完整的邻接矩阵，不是采样过的
    # logger.info('Adj path: ' + adj_path)
    adj_path = '/home/data/xjn/23largest_baseline/LargeST/data/hk/adj_938.npy' # 完整的邻接矩阵，不是采样过的
    logger.info('Adj path: ' + adj_path)
    
    adj_mx = load_adj_from_numpy(adj_path) # 加载原始邻接矩阵 (938,938)

    # 初始化的分布，初始采样，baseline不变，Fairness每个batch改变！
    ptr1 = np.load(os.path.join(data_path, args.years, 'his_initial200.npz'), allow_pickle=True) # his.npz为原始数据 (N,938,ci) ci=1
    sample_list, sample_dict, sample_map = ptr1['sample_list'], ptr1['sample_dict'].item(), ptr1['sample_map'].item() # 字典，加上item()
    # with open('data/sd/sd_district2.json', 'rb') as file: # 打开 JSON 文件
    #     sd_district = pickle.load(file) # 读取文件内容, 字典
    with open('data/hk/district13_roadindex.json', 'r') as file: # 打开 JSON 文件
        district13_road_index = json.load(file) # 读取文件内容, 字典

    district_nodes = get_district_nodes(sample_dict, sample_map) # 初始化的区域字典，键0-12，值list(0-449)
    
    # 读取全数据
    dataloader, scaler = load_dataset2(data_path, args, logger) # dataloader为938

    model = FSAMPLE(node_num=args.sam_num,
                  scaler = scaler,
                  input_dim=args.input_dim,
                  output_dim=args.output_dim,
                  dis_output_dim = args.dis_output_dim,
                  logger=logger,
                  device=device,
                  adj_mx=adj_mx, # 初始的邻接矩阵，938*938，后面会随着采样改变
                  n_filters=args.n_filters, # 类似hidden_size
                  max_diffusion_step=args.max_diffusion_step,
                  filter_type=args.filter_type,
                  num_rnn_layers=args.num_rnn_layers,
                  cl_decay_steps=args.cl_decay_steps
                  )

    # 定义损失函数和优化器
    loss_fn = masked_mae # 损失取mae
    criterion = nn.BCELoss() # 多分类+softmax   nn.CrossEntropyLoss() #  # 鉴别器损失
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    steps = [10, 50, 90]  # CA: [5, 50, 90], others: [10, 50, 90]
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=0.1, verbose=True)
    scheduler = None
    
    engine = FSAMPE_Engine(criterion = criterion,
                          T_dynamic = args.T_dynamic,
                          sample_num = args.sam_num,
                          district13_road_index = district13_road_index,
                        
                          device=device,
                          model=model,
                          dataloader=dataloader,
                          scaler=scaler,
                          sampler=None,
                          loss_fn=loss_fn, # 任务损失mae！
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