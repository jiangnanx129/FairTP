import torch
import numpy as np
import sys
import os
import time
file_path = os.path.abspath(__file__) # /home/data/xjn/8fairness/src/engines/dcrnn_engine.py

from src.base.engine import BaseEngine # special
from src.utils.metrics import masked_mape, masked_rmse
from src.utils.metrics_region import masked_mae3, masked_mape3
from src.utils.metrics import compute_all_metrics
import random
import torch.nn.functional as F
'''
为baseline计算IDF(基于个体的动态损失)
'''
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

'''RSF，基于区域的静态损失计算，type1：'''
# 希望输出是每个区域一个(b,t,1)
def static_cal(pred,label): # (b, t, 13, 1), 13表示区域数量，1表示预测的交通值维度
    b, t = pred.shape[0], pred.shape[1]
    pred = pred.reshape(b * t, -1) # 将 pred 和 label 转换为形状为 (b*t, 13)
    label = label.reshape(b * t, -1)

    # 初始化一个张量用于保存每个区域对之间的 MAPE 差值
    mpe_diff, mape_diff = [],[] # torch.zeros((78,))
    mae_diff = []

    # 计算 MAPE 差值
    idx = 0
    for i in range(pred.shape[1]-1): # 区域数量-1-->13-1=12 # i = 0123456789--11
        mpe_i = (pred[:, i] - label[:, i]) / label[:, i] # 区域i的mpe
        mape_i = torch.abs(mpe_i) # 区域i的mape
        mae_i = torch.abs(pred[:, i] - label[:, i])
        for j in range(i + 1, pred.shape[1]): 
            mpe_j = (pred[:, j] - label[:, j]) / label[:, j] # 区域j的mpe            
            mape_j = torch.abs(mpe_j) # 区域j的mape
            mae_j = torch.abs(pred[:, j] - label[:, j])
            # mape_diff[idx] = torch.mean(mape_i - mape_j)
            # mape_diff.append(torch.abs(torch.sum(mape_i - mape_j))) # mean换成sum 
            mpe_diff.append(torch.sum(torch.abs(mpe_i - mpe_j))) # 区域i和区域j，每个时刻误差的绝对值
            mape_diff.append(torch.sum(torch.abs(mape_i - mape_j))) # sum可换成mean
            mae_diff.append(torch.sum(torch.abs(mae_i - mae_j)))    
            '''sum换成mean'''
            idx += 1
        # print("--------------------:",mape_i, mape_j,mape_j.shape)
        
    mpe_diff_mean = torch.mean(torch.stack(mpe_diff), dim=0)
    mape_diff_mean = torch.mean(torch.stack(mape_diff), dim=0)
    mae_diff_mean = torch.mean(torch.stack(mae_diff), dim=0)

    return mpe_diff_mean, mape_diff_mean, mae_diff_mean

# 不要mask，计算mape和mae
def static_cal2(pred,label): # (b, t, 13, 1), 13表示区域数量，1表示预测的交通值维度
    b, t = pred.shape[0], pred.shape[1]
    pred = pred.reshape(b * t, -1) # 将 pred 和 label 转换为形状为 (b*t, 13)
    label = label.reshape(b * t, -1)

    # 初始化一个张量用于保存每个区域对之间的 MAPE 差值
    mpe_diff, mape_diff = [],[] # torch.zeros((78,))
    mae_diff = []

    # 计算 MAPE 差值
    idx = 0
    for i in range(pred.shape[1]-1): # 区域数量-1-->13-1=12 # i = 0123456789--11
        mpe_i = (pred[:, i] - label[:, i]) / label[:, i] # 区域i的mpe
        mape_i = torch.abs(mpe_i) # 区域i的mape
        mae_i = torch.abs(pred[:, i] - label[:, i])
        for j in range(i + 1, pred.shape[1]): 
            mpe_j = (pred[:, j] - label[:, j]) / label[:, j] # 区域j的mpe            
            mape_j = torch.abs(mpe_j) # 区域j的mape
            mae_j = torch.abs(pred[:, j] - label[:, j])
            # mape_diff[idx] = torch.mean(mape_i - mape_j)
            # mape_diff.append(torch.abs(torch.sum(mape_i - mape_j))) # mean换成sum 
            mpe_diff.append(torch.sum(torch.abs(mpe_i - mpe_j))) # 区域i和区域j，每个时刻误差的绝对值
            mape_diff.append(torch.sum(torch.abs(mape_i - mape_j))) # sum可换成mean
            mae_diff.append(torch.sum(torch.abs(mae_i - mae_j)))    
            '''sum换成mean'''
            idx += 1
        # print("--------------------:",mape_i, mape_j,mape_j.shape)
        
    mpe_diff_mean = torch.mean(torch.stack(mpe_diff), dim=0)
    mape_diff_mean = torch.mean(torch.stack(mape_diff), dim=0)
    mae_diff_mean = torch.mean(torch.stack(mae_diff), dim=0)

    return mpe_diff_mean, mape_diff_mean, mae_diff_mean




def static_cal2_mae(pred,label,mask_value): # (b, t, 13, 1), 13表示区域数量，1表示预测的交通值维度
    region_mae = masked_mae3(pred,label,mask_value)

    count_num = 0
    dis_region = 0
    region_count = pred.shape[2]
    for i in range(region_count-1):
        for j in range(i+1, region_count):
            dis_region += torch.abs(region_mae[i]-region_mae[j])
            count_num +=1  
    
    return dis_region/count_num

def static_cal2_mape(pred,label,mask_value): # (b, t, 13, 1), 13表示区域数量，1表示预测的交通值维度
    region_mae = masked_mape3(pred,label,mask_value)

    count_num = 0
    dis_region = 0
    region_count = pred.shape[2]
    for i in range(region_count-1):
        for j in range(i+1, region_count):
            dis_region += torch.abs(region_mae[i]-region_mae[j])
            count_num +=1  
    
    return dis_region/count_num


def calcu_mape2(preds, labels, null_val): # 均为(b,t,n,c)-gpu, 在region_map_test.py中有测试
    # loss = torch.abs((preds - labels) / labels) # (b,t,n,c)
    # #loss = torch.mean(loss, dim=(3)) # (b,t,n)

    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return loss 

def give_yl_label2(mape_data, yl_values): # yl_values阈值，大于表示误差大判断为l，小于表示误差小判断为y
    # mape_data[mape_data > yl_values] = -1
    # mape_data[mape_data <= yl_values] = 1 # mape_data是(b,n)
    b,t,n,_ = mape_data.shape
    # yl_label = torch.where(mape_data > yl_values, -1, 1) # 误差大就0！表示牺牲
    yl_label = torch.where(mape_data > yl_values, torch.full_like(mape_data, -1), torch.ones_like(mape_data))# .reshape(b,t,n,-1) # 鉴别器输出的标签(t*b*n,co),为-1/1

    return yl_label

# 计算mape，和阈值比较，大于阈值为牺牲，小于阈值为受益
def dynamic_cal(pred, label, mask_value, yl_values, sample_map, district_nodes): # (b,t,n,co)
    mape_data = calcu_mape2(pred, label, mask_value) # 计算每个节点的mape（不分高估/低估）
    yl_label = give_yl_label2(mape_data, yl_values) # 根据mape的误差判断优劣情况，误差小为优，误差大为劣。返回yl_label为(b,t,n,co),全是1和-1
    
    node_yl_dic = {}
    for v, node_list in district_nodes.items(): # 键区域0-12，值list 0-449
        for node in node_list: # (0-449)
            node_938 = sample_map[node]
            if node_938 not in node_yl_dic:
                node_yl_dic[node_938] = 0
            node_yl_dic[node_938] += torch.sum(yl_label[:,:, node, :].flatten()) # 当前节点，在这段时间内的总体牺牲和受益情况。
            '''思考：牺牲和受益是否需要分开看？'''

    # 计算动态损失：只基于采样节点
    differences = []
    node_keys = list(node_yl_dic.keys())  # 这段时间出现过的采样节点，0-938. # 可能len为500(>450) 
    num_nodes = len(node_keys) # 一个batch内就是450
    # 计算节点两两之间的差值，只考虑了采样出来的节点
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            diff_dout = torch.abs(node_yl_dic[node_keys[i]] - node_yl_dic[node_keys[j]]) # 两个值相减
            differences.append(diff_dout) # 基于+1/-1
    
    differences_sum = torch.mean(torch.stack(differences), dim=0) 
    # differences_sum = torch.mean(torch.tensor(differences)) 
    return differences_sum


'''基于dis_out来梯度下降，返回一个follow dis_out,计算的-1/1的值'''
def dynamic_cal5_global3(dis_out, sample_map, district_nodes, district13_road_index, node_count_global):
    b,t,n,c = dis_out.shape
    dis_out = dis_out - 0.5 
    # print("1. dis_out requires_grad:", dis_out.requires_grad)
    dis_labels_new = torch.where(dis_out > 0, torch.ones_like(dis_out), torch.full_like(dis_out, -1))
    
    node_yl_dic_disout = {} # 键为0-938，值为对应的dis_out的和 0.几
    dis_out_cal_dict = {} # 用于计算基于dis_label
      
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
            if node_938 not in node_count_global:
                node_count_global[node_938] = 0
            if node_938 not in dis_out_cal_dict:
                dis_out_cal_dict[node_938] = 0
                
            # node_yl_dic[node_938] += torch.sum(dis_out_list[i][:,node].flatten()) # 应该是一个值，无数个0.几相加，针对digmoid输出
            #node_yl_dic[node_938] += torch.sum(dis_labels[:,:, node, :].flatten()) # 这段时间内的总体的受益/损失的加和，+-1
            '''原来dis_out,取值0-1，dis_label取值1/-1'''
            node_yl_dic_disout[node_938] += torch.sum(dis_out[:,:, node, :].flatten()) # 某个节点这段时间的牺牲和受益总体情况，sum的是sigmoid后-0.5的值，可能大于或小于0
            dis_out_cal_dict[node_938] += torch.sum(dis_labels_new[:,:, node, :].flatten())
            # print("2. node_yl_dic_disout requires_grad:", node_yl_dic_disout[node_938].requires_grad) # T
            node_count_global[node_938] += 1 # 计数，采样过几次
              
    # 计算动态损失：只基于采样节点
    differences_dlabel, differences_dout = [],[]
    node_keys = list(node_yl_dic_disout.keys()) # 这段时间出现过的采样节点，0-938
    num_nodes = len(node_keys) # 可能len为500(>450) # 一个batch内就是450
    # 计算节点两两之间的差值，只考虑了采样出来的节点
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            diff_dout = torch.abs(node_yl_dic_disout[node_keys[i]] - node_yl_dic_disout[node_keys[j]]) # 两个值相减
            diff_dlabel = torch.abs(dis_out_cal_dict[node_keys[i]] - dis_out_cal_dict[node_keys[j]]) 
            # print("1. diff_dout requires_grad:", diff_dout.requires_grad) # T
            differences_dlabel.append(diff_dlabel)
            differences_dout.append(diff_dout)
            
    differences_sum_dlabel = torch.mean(torch.tensor(differences_dlabel), dim=0) # 基于+-1计算出的动态损失
    differences_sum_dout = torch.mean(torch.stack(differences_dout), dim=0) # torch.mean(torch.tensor(differences_dout))没有梯度！！ # 基于dis_out, 0.x计算出的动态损失

    # print("1. differences_sum_dout requires_grad:", differences_sum_dout.requires_grad) # T
    return differences_sum_dout, differences_sum_dlabel, node_count_global

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



















# 计算一段时间的每个节点的牺牲/受益状况，+-1？？dis_labels是+-1，dis_out是0-1之间
def dynamic_cal5(dis_labels, dis_out, sample_map, district_nodes, district13_road_index):
    b,t,n,c = dis_labels.shape
    dis_out = dis_out.reshape(b,t,n,-1)
    # print("原来的输出：", dis_out)
    dis_out = dis_out - 0.5 # torch.where(dis_out > 0.5, dis_out - 0.5, 0.5 - dis_out)#.abs())
    # print("后来的输出：", dis_out)
    
    node_yl_dic = {} 
    node_yl_dic_disout = {} # 键为0-938，值为对应的dis_out的和 0.几

    for v, node_list in district_nodes.items(): # 键区域0-12，值list 0-449
        for node in node_list: # (0-449)
            node_938 = sample_map[node]
            if node_938 not in node_yl_dic_disout:
                #node_yl_dic[node_938] = 0
                node_yl_dic_disout[node_938] = 0
                
            # node_yl_dic[node_938] += torch.sum(dis_out_list[i][:,node].flatten()) # 应该是一个值，无数个0.几相加，针对digmoid输出
            #node_yl_dic[node_938] += torch.sum(dis_labels[:,:, node, :].flatten()) # 这段时间内的总体的受益/损失的加和，+-1
            node_yl_dic_disout[node_938] += torch.sum(dis_out[:,:, node, :].flatten())

    # 计算动态损失：只基于采样节点
    differences_dlabel, differences_dout = [],[]
    node_keys = list(node_yl_dic_disout.keys()) # 这段时间出现过的采样节点，0-938
    num_nodes = len(node_keys) # 可能len为500(>450) # 一个batch内就是450
    # 计算节点两两之间的差值，只考虑了采样出来的节点
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            #diff_dlabel = torch.abs(node_yl_dic[node_keys[i]] - node_yl_dic[node_keys[j]]) # 两个值相减
            diff_dout = torch.abs(node_yl_dic_disout[node_keys[i]] - node_yl_dic_disout[node_keys[j]]) # 两个值相减
            #differences_dlabel.append(diff_dlabel)
            differences_dout.append(diff_dout)
            
    #differences_sum_dlabel = torch.mean(torch.tensor(differences_dlabel)) # 基于+-1计算出的动态损失
    differences_sum_dout = torch.mean(torch.tensor(differences_dout)) # 基于dis_out, 0.x计算出的动态损失

    # softmax算采样概率，不除以64和12，node_yl_dic原始506/-42，node_yl_dic_disout原始26/-2
    sum_node_list = list(district13_road_index.values()) # sum_node_list所有节点
    flat_node_list = [num for sublist in sum_node_list for num in sublist] # 展开2维list,所有节点938
    new_node_yl_dic, new_node_yl_dic_disout = dict(node_yl_dic), dict(node_yl_dic_disout) # node_yl_dic的修改不会影响new_node_yl_dic，它是独立的
    for node_938 in flat_node_list:
        # 针对+-1的
        #if node_938 in new_node_yl_dic: # node_yl_dic中原本就有，该节点在过去采样中出现过
            #continue
        #else:
            #new_node_yl_dic[node_938] = torch.tensor(0.0).to(device) # 没有采样过的节点设为0，不牺牲不收益
        
        # 针对0.x的
        if node_938 in new_node_yl_dic_disout: # node_yl_dic中原本就有，该节点在过去采样中出现过
            continue
        else:
            new_node_yl_dic_disout[node_938] = torch.tensor(0.0, requires_grad=True).to(device)
    

    # sample = []
    #sorted_new_node_yl_dic = dict(sorted(new_node_yl_dic.items(), key=lambda x: x[0])) # 按键排序，0-938
    sorted_new_node_yl_dic_disout = dict(sorted(new_node_yl_dic_disout.items(), key=lambda x: x[0]))
    
    #values_label = torch.stack(list(sorted_new_node_yl_dic.values()))
    #softmax_lvalues = F.softmax(-(values_label/64), dim=0) # (N,)--(938,)是tensor /(64)

    #values_dis = torch.stack(list(sorted_new_node_yl_dic_disout.values()))
    #softmax_dvalues = F.softmax(-(values_dis/64), dim=0)
    #softmax_dvalues = F.sigmoid(-(values_dis/64)) # 0.3976(高受益)，0.5093(大牺牲)，返回938的list
    '''上三行用下两行替换'''
    _,norm_tensor = normalize_list(list(sorted_new_node_yl_dic_disout.values()))
    values_node = torch.sigmoid(norm_tensor)

    return differences_sum_dout, values_node # 基于0.x的loss和list节点采样概率
    # sorted_new_node_yl_dic, sorted_new_node_yl_dic_disout, differences_sum_dlabel, differences_sum_dout, softmax_lvalues, softmax_dvalues
    # +-1字典，0.x字典，Dloss值，Dloss值, softmax的list, softmax的list2