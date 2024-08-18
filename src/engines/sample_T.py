import torch
import numpy as np
import sys
import os
import time
file_path = os.path.abspath(__file__) # /home/data/xjn/8fairness/src/engines/dcrnn_engine.py
# print(file_path)
# current_directory = os.getcwd()
# print(current_directory)

# from src.base.engine import BaseEngine # special
# from src.utils.metrics import masked_mape, masked_rmse
# from src.utils.metrics import compute_all_metrics
import random
from src.utils.metrics_region import masked_mae3, masked_mape3
import torch.nn.functional as F
'''
专门放 两个公平的计算函数，以及采样所需要的函数
'''
from collections import Counter

# def merge_dicts(dict1, dict2, dict3):
#     merged_dict = Counter(dict1) + Counter(dict2) + Counter(dict3)
#     return dict(merged_dict)

def merge_dicts(*dicts):
    counters = [Counter(d) for d in dicts]
    merged_counter = sum(counters, Counter())
    merged_dict = dict(merged_counter)
    return merged_dict


# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# 希望输出是每个区域一个(b,t,1)
def static_cal(pred,label, device): # (b, t, 13, 1), 13表示区域数量，1表示预测的交通值维度
    b, t = pred.shape[0], pred.shape[1]
    pred = pred.reshape(b * t, -1) # 将 pred 和 label 转换为形状为 (b*t, 13)
    label = label.reshape(b * t, -1)

    mape_diff = [] # torch.zeros((78,))
    region_mape_list = [] # 最终长度为13, 每个区域的mape，值大误差大 要多数采样
    # 计算 MAPE 差值
    idx = 0
    for i in range(pred.shape[1]-1): # 区域数量-1-->13-1=12 # i = 0123456789--11
        mape_i = torch.abs(pred[:, i] - label[:, i]) / label[:, i] # 区域i的mape
        region_mape_list.append(1/torch.sum(mape_i))
        for j in range(i + 1, pred.shape[1]):
            mape_j = torch.abs(pred[:, j] - label[:, j]) / label[:, j] # 区域j的mape
            # mape_diff[idx] = torch.mean(mape_i - mape_j)
            # mape_diff.append(torch.abs(torch.sum(mape_i - mape_j))) # mean换成sum 
            mape_diff.append(torch.sum(torch.abs(mape_i - mape_j))) # 此处sum加的是b*t的时间维度
            idx += 1

    mape_diff_mean = torch.mean(torch.stack(mape_diff), dim=0)
    region_mape_list.append(1/(torch.sum(torch.abs(pred[:, -1] - label[:, -1]) / label[:, -1]))) # 加上最后一个区域的概率，mape

    '''1/：误差越大，1/越小，我们的目标是，多采样误差大的，也就是找sigmoid后小的！！！'''
    # region_mape_list中元素6百到2000
    _,norm_tensor = normalize_list(region_mape_list) # 归一化
    values_region = torch.sigmoid(norm_tensor).to(device) # 输出限定在0到1之间，将输出解释为正类的概率（二分类）

    return mape_diff_mean, values_region # each为长为13的list 对应0-12个区域


# 希望输出是每个区域一个(b,t,1)
def static_cal2_mae(pred,label,mask_value, device): # (b, t, 13, 1), 13表示区域数量，1表示预测的交通值维度
    region_mae = masked_mae3(pred,label,mask_value) # 返回[13]
    count_num = 0
    dis_region = 0
    region_count = pred.shape[2]
    for i in range(region_count-1):
        for j in range(i+1, region_count):
            dis_region += torch.abs(region_mae[i]-region_mae[j])
            count_num +=1

    '''1/：误差越大，1/越小，我们的目标是，多采样误差大的，也就是找sigmoid后小的！！！'''
    # region_mape_list中元素6百到2000
    values_region = torch.sigmoid(region_mae).to(device) # 输出限定在0到1之间，将输出解释为正类的概率（二分类）

    return dis_region/count_num, values_region # each为长为13的list 对应0-12个区域


def static_cal2_mape(pred,label,mask_value, device): # (b, t, 13, 1), 13表示区域数量，1表示预测的交通值维度
    region_mae = masked_mape3(pred,label,mask_value) # 返回[13]
    count_num = 0
    dis_region = 0
    region_count = pred.shape[2]
    for i in range(region_count-1):
        for j in range(i+1, region_count):
            dis_region += torch.abs(region_mae[i]-region_mae[j])
            count_num +=1

    '''1/：误差越大，1/越小，我们的目标是，多采样误差大的，也就是找sigmoid后小的！！！'''
    # region_mape_list中元素6百到2000
    values_region = torch.sigmoid(1/region_mae).to(device) # 输出限定在0到1之间，将输出解释为正类的概率（二分类）

    return dis_region/count_num, values_region # each为长为13的list 对应0-12个区域


# 基于mape确定的阈值，用阈值和mape比较得到的label，get_yl_batch. 实际输入是鉴别其输出dis_out: (450,)
'''方便考虑1234'''
def dynamic_cal2(time_T, yl_node_dic_list, district13_road_index, device):
    dict1, dict2, dict3 = yl_node_dic_list[0], yl_node_dic_list[1], yl_node_dic_list[2]
    node_dis_dic = merge_dicts(dict1, dict2, dict3)
    # T段时间总共的节点状态，我的目标是找出最小的

    # 计算动态损失：只基于采样节点
    differences_dout = []
    node_keys = list(node_dis_dic.values()) # 这段时间出现过的采样节点，0-938
    num_nodes = len(node_keys) # 可能len为500(>450) # 一个batch内就是450
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            diff_dout = torch.abs(node_keys[i] - node_keys[j]) # 两个值相减
            differences_dout.append(diff_dout)
            
    differences_sum_dout = torch.mean(torch.stack(differences_dout), dim=0) # torch.mean(torch.tensor(differences_dout))没有梯度！！ # 基于dis_out, 0.x计算出的动态损失
    
    '''直觉：采样牺牲的节点，node_dis_dic保存的是节点状态，直接要最小的'''
    sum_node_list = list(district13_road_index.values()) # sum_node_list所有节点
    flat_node_list = [num for sublist in sum_node_list for num in sublist] # 展开2维list,所有节点938
    new_node_yl_dic_disout = {key: value for key, value in node_dis_dic.items()}
    for node_938 in flat_node_list:
        if node_938 in new_node_yl_dic_disout: # node_yl_dic中原本就有，该节点在过去采样中出现过
            continue
        else:
            new_node_yl_dic_disout[node_938] = torch.tensor(0.0, requires_grad=True).to(device)
    
    sorted_new_node_yl_dic_disout = dict(sorted(new_node_yl_dic_disout.items(), key=lambda x: x[0]))
    _,norm_tensor = normalize_list(list(sorted_new_node_yl_dic_disout.values()))
    values_node = torch.sigmoid(norm_tensor)

    return differences_sum_dout, values_node


def get_district_nodes(sample_dict, sample_map): # 2字典，sample_dict键0-12，值list(0-937); sample_map键0-449,值(0-938)
    district_nodes = {}
    for v, node_list in sample_dict.items(): # 一定有0-12，但是select_list可能为空：某个区域没有采样节点！会导致engine的new_pred报错(102行)
        select_list  = [key for key, value in sample_map.items() if value in node_list]
        district_nodes[v] = select_list # 保存每个区域要取的节点（0-449），键为区域（0-12），值为list
    return district_nodes # 字典，键0-12，值list(0-449)

# 将sample出的列表与450对应，sample为长度为450的list，其中每个元素取值(0-937)
def sum_map(sample, sam_num):
    sample_map = {}
    for i in range(sam_num):
        sample_map[i] = sample[i]
    return sample_map # 字典，键为450个节点的下标(取值0-449)，值为对应的节点下标（取值0-937）

# 采样的节点map到几个区域
def sample_map_district(sample_list, district13_road_index):
    # 生成新的字典来存储节点和其所属的区域信息
    new_dict = {} # 键为区域下标（0-12），值为区域对应节点列表

    # 遍历原始字典
    for district_id, nodes in district13_road_index.items():
        for node in nodes: # nodes为list, 循环938个节点！
            if node in sample_list:
                if district_id not in new_dict: # 返回sample_dict, 没有采样的区域就没有该区域id，0-12可能缺少7
                    new_dict[district_id] = []
                new_dict[district_id].append(node)

    # print(new_dict)
    return new_dict # 每个值一定是从小到大排列的！


# 归一化
def normalize_list(input_list):
    # 将列表中的张量转换为一个大张量
    tensor_list = torch.stack(input_list)

    # 计算最小值和最大值
    min_val = torch.min(tensor_list)
    max_val = torch.max(tensor_list)

    # 归一化处理
    normalized_list = [(x - min_val) / (max_val - min_val) for x in input_list]
    normalized_tensor = (tensor_list - min_val) / (max_val - min_val)

    return normalized_list, normalized_tensor






def count_values(dictionary):
    values = torch.tensor(list(dictionary.values()))  # 将值转换为张量

    greater_than_zero = torch.sum(values > 0.0)
    less_than_zero = torch.sum(values < 0.0)
    equal_to_zero = torch.sum(values == 0.0)

    return greater_than_zero.item(), less_than_zero.item(), equal_to_zero.item()







# 计算的loss4是根据0的得到的，要求都距离0比较近
'''
每次采样都算，只有T次才参与backward
node_yl_dic_disout是全局的字典，每T次清空一次
'''
def dynamic_cal5_global_no5_3(time_T_cal, dis_labels, dis_out, sample_map, district_nodes, node_yl_dic_disout): # , district13_road_index, node_count_global):
    b,t,n,c = dis_labels.shape
    dis_out = dis_out.reshape(b,t,n,-1) # linear2后经过sigmoid，取值0-1表示受益/牺牲的状态
    dis_out = dis_out - 0.5 # torch.where(dis_out > 0.5, dis_out - 0.5, 0.5 - dis_out)#.abs())
    
    node_yl_dic = {} 
    
    '''
    如果每个batch执行一次，其实每次都只有450个节点，没有考虑之前采样的情况，影响366，不可能大于450
    考虑全局情况，应该要设置一个全局的node_yl_dic_disout，而不是每次={}
    '''

    for v, node_list in district_nodes.items(): # 键区域0-12，值list 0-449
        for node in node_list: # (0-449)
            node_938 = sample_map[node]
            if node_938 not in node_yl_dic_disout:
                #node_yl_dic[node_938] = 0
                node_yl_dic_disout[node_938] = 0
            '''采样的时候算过了''' # sum表示节点状态
            node_yl_dic_disout[node_938] += torch.sum(dis_out[:,:, node, :].flatten()) # 某个节点这段时间的牺牲和受益总体情况，sum的是sigmoid后-0.5的值，可能大于或小于0

    '''统计优劣的比例'''
    greater_than_zero, less_than_zero, equal_to_zero = count_values(node_yl_dic_disout) # 总体情况。每个节点在一段时间内的状态
    
    if  time_T_cal:
        # 计算动态损失：只基于采样节点
        differences_dlabel, differences_dout = [],[]
        node_keys = list(node_yl_dic_disout.keys()) # 这段时间出现过的采样节点，0-938
        # print("a:",len(list(node_yl_dic_disout.keys())))
        num_nodes = len(node_keys) # 可能len为500(>450) # 一个batch内就是450
        # 计算节点两两之间的差值，只考虑了采样出来的节点
        for i in range(num_nodes - 1):
            for j in range(i + 1, num_nodes):
                #diff_dlabel = torch.abs(node_yl_dic[node_keys[i]] - node_yl_dic[node_keys[j]]) # 两个值相减
                diff_dout = torch.abs(node_yl_dic_disout[node_keys[i]] - node_yl_dic_disout[node_keys[j]]) # 两个值相减
                # print("1. diff_dout requires_grad:", diff_dout.requires_grad) # T
                #differences_dlabel.append(diff_dlabel)
                differences_dout.append(diff_dout)
            
        differences_sum_dout = torch.mean(torch.stack(differences_dout), dim=0) # torch.mean(torch.tensor(differences_dout))没有梯度！！ # 基于dis_out, 0.x计算出的动态损失
        return differences_sum_dout, greater_than_zero, less_than_zero, equal_to_zero, node_yl_dic_disout
    else:
        # print("b:",len(list(node_yl_dic_disout.keys())))
        return greater_than_zero, less_than_zero, equal_to_zero, node_yl_dic_disout


def dynamic_cal5_global_no5_T(time_T, T_district_nodes, T_sample_map, T_disout): # , district13_road_index, node_count_global):
    b,t,n,c = T_disout[0].shape
    node_dis_dic = {}
    for i in range(time_T): # 012
        T_disout[i] = T_disout[i] - 0.5
        for v, node_list in T_district_nodes[i].items(): # 键区域0-12，值list 0-449
            for node in node_list: # (0-449)
                node_938 = T_sample_map[i][node]
                if node_938 not in node_dis_dic:
                    node_dis_dic[node_938] = 0 # 938
            
                node_dis_dic[node_938] += torch.sum(T_disout[i][:,:,node,:].flatten())
    
    # 计算动态损失：只基于采样节点
    differences_dlabel, differences_dout = [],[]
    node_keys = list(node_dis_dic.keys()) # 这段时间出现过的采样节点，0-938
    # print("a:",len(list(node_yl_dic_disout.keys())))
    num_nodes = len(node_keys) # 可能len为500(>450) # 一个batch内就是450
    # 计算节点两两之间的差值，只考虑了采样出来的节点
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            #diff_dlabel = torch.abs(node_yl_dic[node_keys[i]] - node_yl_dic[node_keys[j]]) # 两个值相减
            diff_dout = torch.abs(node_dis_dic[node_keys[i]] - node_dis_dic[node_keys[j]]) # 两个值相减
            # print("1. diff_dout requires_grad:", diff_dout.requires_grad) # T
            #differences_dlabel.append(diff_dlabel)
            differences_dout.append(diff_dout)
            
    differences_sum_dout = torch.mean(torch.stack(differences_dout), dim=0) # torch.mean(torch.tensor(differences_dout))没有梯度！！ # 基于dis_out, 0.x计算出的动态损失
    
    return differences_sum_dout


def dynamic_cal5_global_no5_T2(dis_labels, dis_out, sample_map, district_nodes): # , district13_road_index, node_count_global):
    b,t,n,c = dis_labels.shape
    dis_out = dis_out.reshape(b,t,n,-1) # linear2后经过sigmoid，取值0-1表示受益/牺牲的状态
    dis_out = dis_out - 0.5 # torch.where(dis_out > 0.5, dis_out - 0.5, 0.5 - dis_out)#.abs())
    
    node_yl_dic = {} 
    node_yl_dic_disout={}
    
    '''
    如果每个batch执行一次，其实每次都只有450个节点，没有考虑之前采样的情况，影响366，不可能大于450
    考虑全局情况，应该要设置一个全局的node_yl_dic_disout，而不是每次={}
    '''

    for v, node_list in district_nodes.items(): # 键区域0-12，值list 0-449
        for node in node_list: # (0-449)
            node_938 = sample_map[node]
            if node_938 not in node_yl_dic_disout:
                #node_yl_dic[node_938] = 0
                node_yl_dic_disout[node_938] = 0
            '''采样的时候算过了''' # sum表示节点状态
            node_yl_dic_disout[node_938] += torch.sum(dis_out[:,:, node, :].flatten()) # 某个节点这段时间的牺牲和受益总体情况，sum的是sigmoid后-0.5的值，可能大于或小于0

   
    # 计算动态损失：只基于采样节点
    differences_dlabel, differences_dout = [],[]
    node_keys = list(node_yl_dic_disout.keys()) # 这段时间出现过的采样节点，0-938
    # print("a:",len(list(node_yl_dic_disout.keys())))
    num_nodes = len(node_keys) # 可能len为500(>450) # 一个batch内就是450
    # 计算节点两两之间的差值，只考虑了采样出来的节点
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            #diff_dlabel = torch.abs(node_yl_dic[node_keys[i]] - node_yl_dic[node_keys[j]]) # 两个值相减
            diff_dout = torch.abs(node_yl_dic_disout[node_keys[i]] - node_yl_dic_disout[node_keys[j]]) # 两个值相减
            # print("1. diff_dout requires_grad:", diff_dout.requires_grad) # T
            #differences_dlabel.append(diff_dlabel)
            differences_dout.append(diff_dout)
            
    differences_sum_dout = torch.mean(torch.stack(differences_dout), dim=0) # torch.mean(torch.tensor(differences_dout))没有梯度！！ # 基于dis_out, 0.x计算出的动态损失
        
    return differences_sum_dout


def get_yl_batch(dis_out, sample_map): # sample_map键0-449，值0-938
    dis_out = dis_out - 0.5 # torch.where(dis_out > 0.5, dis_out - 0.5, 0.5 - dis_out)#.abs())
    
    yl_node_dic = {}
    for node, node_938 in sample_map.items():
        if node_938 not in yl_node_dic:
            yl_node_dic[node_938] = 0
        yl_node_dic[node_938] += dis_out[node]
    
    return yl_node_dic


def dynamic_cal5_global_T(time_T, yl_node_dic_list): # , district13_road_index, node_count_global):
    
    # node_dis_dic = {}
    dict1, dict2, dict3 = yl_node_dic_list[0], yl_node_dic_list[1], yl_node_dic_list[2]
    node_dis_dic = merge_dicts(dict1, dict2, dict3)
    
    # node_dis_dic = merge_dicts(yl_node_dic_list)
    
    # 计算动态损失：只基于采样节点
    differences_dout = []
    node_keys = list(node_dis_dic.values()) # 这段时间出现过的采样节点，0-938
    num_nodes = len(node_keys) # 可能len为500(>450) # 一个batch内就是450
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            diff_dout = torch.abs(node_keys[i] - node_keys[j]) # 两个值相减
            differences_dout.append(diff_dout)
            
    differences_sum_dout = torch.mean(torch.stack(differences_dout), dim=0) # torch.mean(torch.tensor(differences_dout))没有梯度！！ # 基于dis_out, 0.x计算出的动态损失
    
    return differences_sum_dout
    