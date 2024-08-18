import os
import argparse
import numpy as np

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

from src.models.stgode import STGODE
from src.engines.stgode_engine import STGODE_Engine
# from src.base.engine import BaseEngine
from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger
from fastdtw import fastdtw

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()
    parser.add_argument('--tpd', type=int, default=96, help='time per day')
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--thres', type=float, default=0.6)

    parser.add_argument('--lrate', type=float, default=2e-3)
    parser.add_argument('--wdecay', type=float, default=0)
    parser.add_argument('--clip_grad_value', type=float, default=0)
    parser.add_argument('--yl_values', type=float, default=0.05) # 优劣标签
    parser.add_argument('--sam_num', type=int, default=450) # 
    args = parser.parse_args()

    log_dir = './experiments/{}/{}/'.format(args.model_name, args.dataset)
    logger = get_logger(log_dir, __name__, 'record_s{}_HK_2.log'.format(args.seed))
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
    # 'hk/202010/adj_initial450_all1.npy'

    adj_mx = load_adj_from_numpy(adj_path) # load adjacent
    adj_mx = adj_mx - np.eye(args.sam_num) # 450
    sp_matrix = adj_mx + np.transpose(adj_mx)
    sp_matrix = normalize_adj_mx(sp_matrix).to(device)
    
    se_matrix = construct_se_matrix(data_path, args) # 要读取HK数据的npz文件， initial450
    se_matrix = normalize_adj_mx(se_matrix).to(device)

    print(sp_matrix.shape, se_matrix.shape)
    
    # 初始化的分布，初始采样，baseline不变，Fairness每个batch改变！
    ptr1 = np.load(os.path.join(data_path, args.years, 'his_initial450.npz'), allow_pickle=True) # his.npz为原始数据 (N,938,ci) ci=1
    sample_list, sample_dict, sample_map = ptr1['sample_list'], ptr1['sample_dict'].item(), ptr1['sample_map'].item() # 字典，加上item()
    # with open('data/sd/sd_district.json', 'rb') as file: # 打开 JSON 文件
    #     sd_district = pickle.load(file) # 读取文件内容, 字典
    
    district_nodes = get_district_nodes(sample_dict, sample_map) # 初始化的区域字典，键0-12，值list(0-449)


    dataloader, scaler = load_dataset(data_path, args, logger)

    model = STGODE(node_num=args.sam_num,
                   input_dim=args.input_dim,
                   output_dim=args.output_dim,
                   A_sp=sp_matrix,
                   A_se=se_matrix
                   )

    loss_fn = masked_mae
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    engine = STGODE_Engine(device=device,
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
        engine.train(args.yl_values, sample_map, district_nodes)
    else:
        engine.evaluate(args.mode, args.yl_values, sample_map, district_nodes, epoch=1)


def construct_se_matrix(data_path, args):
    ptr = np.load(os.path.join(data_path, args.years, 'his_initial450.npz')) # hk450
    data = ptr['data'][..., 0]
    sample_num, node_num = data.shape

    data_mean = np.mean([data[args.tpd * i: args.tpd * (i + 1)] for i in range(sample_num // args.tpd)], axis=0)
    data_mean = data_mean.T
    
    dist_matrix = np.zeros((node_num, node_num))
    for i in range(node_num):
        for j in range(i, node_num):
            dist_matrix[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]

    for i in range(node_num):
        for j in range(i):
            dist_matrix[i][j] = dist_matrix[j][i]

    mean = np.mean(dist_matrix)
    std = np.std(dist_matrix)
    dist_matrix = (dist_matrix - mean) / std
    dist_matrix = np.exp(-dist_matrix ** 2 / args.sigma ** 2)
    dtw_matrix = np.zeros_like(dist_matrix)
    dtw_matrix[dist_matrix > args.thres] = 1
    return dtw_matrix


def normalize_adj_mx(adj_mx):
    alpha = 0.8
    D = np.array(np.sum(adj_mx, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), adj_mx),
                         diag.reshape((1, -1)))
    A_reg = alpha / 2 * (np.eye(adj_mx.shape[0]) + A_wave)
    return torch.from_numpy(A_reg.astype(np.float32))


if __name__ == "__main__":
    main()