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
import torch.nn.functional as F
'''
专门放 两个公平的计算函数，以及采样所需要的函数
'''
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


'''采样优势区域，采样误差小的区域，去掉1/'''
def static_cal_2(pred,label, device): # (b, t, 13, 1), 13表示区域数量，1表示预测的交通值维度
    b, t = pred.shape[0], pred.shape[1]
    pred = pred.reshape(b * t, -1) # 将 pred 和 label 转换为形状为 (b*t, 13)
    label = label.reshape(b * t, -1)

    mape_diff = [] # torch.zeros((78,))
    region_mape_list = [] # 最终长度为13, 每个区域的mape，值大误差大 要多数采样
    # 计算 MAPE 差值
    idx = 0
    for i in range(pred.shape[1]-1): # 区域数量-1-->13-1=12 # i = 0123456789--11
        mape_i = torch.abs(pred[:, i] - label[:, i]) / label[:, i] # 区域i的mape
        region_mape_list.append(torch.sum(mape_i))
        for j in range(i + 1, pred.shape[1]):
            mape_j = torch.abs(pred[:, j] - label[:, j]) / label[:, j] # 区域j的mape
            # mape_diff[idx] = torch.mean(mape_i - mape_j)
            # mape_diff.append(torch.abs(torch.sum(mape_i - mape_j))) # mean换成sum 
            mape_diff.append(torch.sum(torch.abs(mape_i - mape_j))) # 此处sum加的是b*t的时间维度
            idx += 1

    mape_diff_mean = torch.mean(torch.stack(mape_diff), dim=0)
    region_mape_list.append((torch.sum(torch.abs(pred[:, -1] - label[:, -1]) / label[:, -1]))) # 加上最后一个区域的概率，mape

    '''1/：误差越大，1/越小，我们的目标是，多采样误差大的，也就是找sigmoid后小的！！！'''
    # region_mape_list中元素6百到2000
    _,norm_tensor = normalize_list(region_mape_list) # 归一化
    values_region = torch.sigmoid(norm_tensor).to(device) # 输出限定在0到1之间，将输出解释为正类的概率（二分类）

    return mape_diff_mean, values_region # each为长为13的list 对应0-12个区域



def dynamic_cal2(dis_labels, sample_map, district_nodes): # dis_out_list必须是(b,t,n)/(b*t,n)d
    
    node_yl_dic = {}
    for v, node_list in district_nodes.items(): # 键区域0-12，值list 0-449
        for node in node_list: # (0-449)
            node_938 = sample_map[node]
            if node_938 not in node_yl_dic:
                node_yl_dic[node_938] = 0
                
            # node_yl_dic[node_938] += torch.sum(dis_out_list[i][:,node].flatten()) # 应该是一个值，无数个0.几相加，针对digmoid输出
            node_yl_dic[node_938] += torch.sum(dis_labels[:,:, node, :].flatten()) # 先除以，再相加，b*t个数相加

    # 计算loss4，后面要加东西了
    differences = []
    node_keys = list(node_yl_dic.keys()) # 这段时间出现过的采样节点，0-938
    num_nodes = len(node_keys) # 可能len为500(>450) # 一个batch内就是450
    # 计算节点两两之间的差值，只考虑了采样出来的节点
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            diff = torch.abs(node_yl_dic[node_keys[i]] - node_yl_dic[node_keys[j]]) # 两个值相减
            differences.append(diff)
    differences_sum = torch.mean(torch.tensor(differences))

    return differences_sum, node_yl_dic 


# 不用计算区域，直接基于单个节点状态采样(node_yl_dic)
def dynamic_sample4(node_yl_dic, district13_road_index, sam_num):
    
    '''
    node_yl_dic: 字典，每个节点这段时间(3个batch)的总体的情况(受益/牺牲融合)
    node_yl_dic2：字典，存储每个节点这段时间(3个batch)的情况-list形式，若有节点出现两次，有长度为2的list，每个list存储其对应时刻的情况
    '''

    sum_node_list = list(district13_road_index.values()) # sum_node_list所有节点
    flat_node_list = [num for sublist in sum_node_list for num in sublist] # 展开2维list

    # 直接就单个节点采样
    new_node_yl_dic = dict(node_yl_dic) # node_yl_dic的修改不会影响new_node_yl_dic，它是独立的
    for node_938 in flat_node_list:
        if node_938 in new_node_yl_dic: # node_yl_dic中原本就有，该节点在过去采样中出现过
            continue
        else:
            new_node_yl_dic[node_938] = 0
    

    sample = []
    sorted_new_node_yl_dic = dict(sorted(new_node_yl_dic.items(), key=lambda x: x[1])) # 键0-937，值yl值，根据值的大小，从小到大重新排序。
    low_keys = [key for key, value in sorted_new_node_yl_dic.items() if value < 0] # 取所有<0
    ban_keys = [key for key, value in sorted_new_node_yl_dic.items() if value == 0] # 取所有=0
    hig_keys = [key for key, value in sorted_new_node_yl_dic.items() if value > 0] # 取所有>0
    
    if sam_num <= len(low_keys): 
        sample.append(low_keys[:sam_num])
    elif (sam_num - len(low_keys)) <= len(ban_keys): # 负数不够, 还要采样的0也不够永
        sample.append(low_keys) # 采所有负数
        nodes_sample = random.sample(ban_keys, (sam_num - len(low_keys))) # 采样数可能超出区域节点数
        sample.append(nodes_sample)
    else: # 负数 + 0 采样也不够
        sample.append(low_keys)
        sample.append(ban_keys)
        remain_s = sam_num- len(low_keys)- len(ban_keys)
        nodes_sample = sample.append(hig_keys[:remain_s])

    sample_list = [item for sublist in sample for item in sublist] # 双层list展开为单层list

    # 判断是否每个区域都有采样
    region_samples = {region: 0 for region in district13_road_index.keys()} # 0-12区域
    new_regions= {} # 键区域0-12，值list采样节点0-938
    for region, node_list in district13_road_index.items(): # 键0-12，值list,(0-937) 
        node_count = 0
        region_to_list = []
        for node_938 in node_list:
            if node_938 in sample_list: # 该区域有节点采样
                region_to_list.append(node_938)
                node_count += 1
                
        region_samples[region] = node_count # 记录该区域的节点采样数目
        new_regions[region] = region_to_list 

    zero_regions = [key for key, value in region_samples.items() if value == 0]
    # 在对应的zero_regions至少采样一个！
    add_node_count = 0
    for region in zero_regions:
        region_nodes = [node for node in list(district13_road_index[region])] # 0-938
        selected_nodes = sorted([node for node in region_nodes if node in sorted_new_node_yl_dic], key=lambda x: sorted_new_node_yl_dic[x])
        sample_list.extend(selected_nodes[:1]) 
        add_node_count += 1 
    if add_node_count != 0:
        # 去掉多采样的！找到最多的区域，在减少
        key_with_max_value = max(region_samples, key=lambda k: region_samples[k]) # 采样最多的区域
        region_nodes = [node for node in list(district13_road_index[key_with_max_value])] # 采样最多的区域的原始节点
        selected_nodes = sorted([node for node in region_nodes if node in sorted_new_node_yl_dic], key=lambda x: sorted_new_node_yl_dic[x])
        new_regions[key_with_max_value]  = sorted(new_regions[key_with_max_value], key=lambda x: selected_nodes.index(x))
        delete_node = new_regions[key_with_max_value][-add_node_count:]
        sample_list = [x for x in sample_list if x not in delete_node]

    sample_map = sum_map(sample_list, sam_num)
    sample_dict = sample_map_district(sample_list, district13_road_index)
    district_nodes = get_district_nodes(sample_dict, sample_map)


    return sample_list, sample_map, sample_dict, district_nodes # 字典，键(0-937),值对应节点的优劣计数




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


def pro_sample(region_mape_list, allnode_list, district13_road_index):
    norm_region,_ = normalize_list(region_mape_list)
    norm_allnode,_ = normalize_list(allnode_list)

    node_cs = {}
    for v,node_list in district13_road_index.items():
        for node in node_list: # 循环该区域的节点列表：
            # node_cs[node] = 0.001*mape_for_each_region[int(v)] * condition_for_each_node[node]
            node_cs[node] = norm_region[int(v)] * norm_allnode[node]

    sorted_dict = dict(sorted(node_cs.items(), key=lambda x: x[0]))
    return sorted_dict

# 都在0.5附近，输入均为tensor(938,)
def pro_sample2(values_region, values_node, district13_road_index):
    node_cs = {}
    for v,node_list in district13_road_index.items(): # 键0-12，值0-938
        for node in node_list: # 循环该区域的节点列表：
            # node_cs[node] = 0.001*mape_for_each_region[int(v)] * condition_for_each_node[node]
            node_cs[node] = values_region[int(v)] * values_node[node]

    sorted_dict = dict(sorted(node_cs.items(), key=lambda x: x[0]))
    # print(list(sorted_dict.values()), len(list(sorted_dict.values()))) # 0.3277
    return sorted_dict

# 采样不考虑区域静态
def pro_sample3(values_node, district13_road_index):
    node_cs = {}
    for v,node_list in district13_road_index.items(): # 键0-12，值0-938
        for node in node_list: # 循环该区域的节点列表：
            # node_cs[node] = 0.001*mape_for_each_region[int(v)] * condition_for_each_node[node]
            node_cs[node] = values_node[node]

    sorted_dict = dict(sorted(node_cs.items(), key=lambda x: x[0]))
    # print(list(sorted_dict.values()), len(list(sorted_dict.values()))) # 0.3277
    return sorted_dict

# 采样不考虑动态区域
def pro_sample4(values_region, district13_road_index):
    node_cs = {}
    for v,node_list in district13_road_index.items(): # 键0-12，值0-938
        for node in node_list: # 循环该区域的节点列表：
            # node_cs[node] = 0.001*mape_for_each_region[int(v)] * condition_for_each_node[node]
            node_cs[node] = values_region[int(v)] 

    sorted_dict = dict(sorted(node_cs.items(), key=lambda x: x[0]))
    # print(list(sorted_dict.values()), len(list(sorted_dict.values()))) # 0.3277
    return sorted_dict

def resampel(sorted_dict, sam_num, district13_road_index):
    sample = []
    for v,node_list in district13_road_index.items():
        values_list = [sorted_dict[key] for key in node_list]
        max_key = node_list[values_list.index(max(values_list))]
        sample.append(max_key)
        
    keys_tensor = torch.tensor(list(sorted_dict.keys()))
    tensor_values = torch.stack(list(sorted_dict.values()))
    total_sum = tensor_values.sum()
    normalized_values = tensor_values / total_sum # 和为1

    # # 使用torch.multinomial()函数进行随机选择
    # sampled_indices = torch.multinomial(normalized_tensor, sam_num, replacement=False)
    mask = torch.ones_like(normalized_values, dtype=torch.bool)
    mask[sample] = False
    new_tensor = torch.masked_select(normalized_values, mask)
    sampled_indices = torch.multinomial(new_tensor, sam_num-len(sample), replacement=False)

    keys_tensor = keys_tensor.to(sampled_indices.device)
    sampled_keys = keys_tensor[sampled_indices]
    sampled_list = sampled_keys.tolist()
    # print(list11.device)

    # sampled_list = np.random.choice(list11, size=sam_num-len(sample), p=list22.cpu())
    return sample+sampled_list


# 没有list，一个batch一采样
def dynamic_cal(dis_out, dis_labels, sample_map, district_nodes, node_nums): # dis_out_list必须是(b,t,n)/(b*t,n)d
    # node_yl_dic = {} # 字典，键0-937，值优劣计数，不一定有938，可能有节点一直没被采样过
    # no_sample = torch.full_like(dis_out[:,1], torch.tensor(2.0, requires_grad=True, device="cuda:0") ) # 
    no_sample = torch.ones_like(dis_out[:, 1], device="cuda:0") * 2.0
    no_sample.requires_grad_(True)
    allnode_yl_dic = {key: torch.sum(no_sample.flatten()) for key in range(node_nums)} # node_nums是节点数目938， 
    
    node_yl_dic, node_yl_dic01 = {},{}
    for v, node_list in district_nodes.items(): # 键区域0-12，值list 0-449
        for node in node_list: # (0-449)
            node_938 = sample_map[node]
            if node_938 not in node_yl_dic:
                node_yl_dic[node_938] = 0
                node_yl_dic01[node_938] = 0
            # node_yl_dic[node_938] += torch.sum(dis_out_list[i][:,node].flatten()) # 应该是一个值，无数个0.几相加，针对digmoid输出
            node_yl_dic[node_938] += torch.sum((1/dis_out[:,node]).flatten()) # 先除以，再相加，b*t个数相加
            node_yl_dic01[node_938] += torch.sum(dis_labels[:,:, node, :].flatten()) # dis_labels-(b,t,n,1)
                
    allnode_yl_dic.update(node_yl_dic) # 用node_yl_dic中的内容更新allnode
    allnode_list = list(allnode_yl_dic.values()) # 对应0-938

    # 计算loss4，后面要加东西了
    differences = []
    node_keys = list(node_yl_dic01.keys())
    num_nodes = len(node_keys) # 可能len为500(>450)
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            diff = torch.abs(node_yl_dic01[node_keys[i]] - node_yl_dic01[node_keys[j]])
            differences.append(diff)
    differences_sum = torch.mean(torch.tensor(differences))

    return differences_sum, node_yl_dic, allnode_yl_dic, allnode_list 


# # 有list版本
# def dynamic_cal2(dis_out_list, sample_map_list, district_nodes_list, node_nums): # dis_out_list必须是(b,t,n)/(b*t,n)d
#     # node_yl_dic = {} # 字典，键0-937，值优劣计数，不一定有938，可能有节点一直没被采样过
#     no_sample = torch.full_like(dis_out_list[0][:,1], torch.tensor(2.0, requires_grad=True, device="cuda:0") ) # 
#     allnode_yl_dic = {key: len(dis_out_list)*torch.sum(no_sample.flatten()) for key in range(node_nums)} # node_nums是节点数目938， 
    
#     node_yl_dic = {}
#     for i in range(len(dis_out_list)):
#         for v, node_list in district_nodes_list[i].items(): # 键区域0-12，值list 0-449
#             for node in node_list: # (0-449)
#                 node_938 = sample_map_list[i][node]
#                 if node_938 not in node_yl_dic:
#                     node_yl_dic[node_938] = 0
#                 # node_yl_dic[node_938] += torch.sum(dis_out_list[i][:,node].flatten()) # 应该是一个值，无数个0.几相加，针对digmoid输出
#                 node_yl_dic[node_938] += torch.sum((1/dis_out_list[i][:,node]).flatten()) # 先除以，再相加，b*t个数相加
                

#     allnode_yl_dic.update(node_yl_dic) # 用node_yl_dic中的内容更新allnode
#     allnode_list = list(allnode_yl_dic.values()) # 对应0-938

#     return node_yl_dic, allnode_yl_dic, allnode_list 

'''计算字典的值之中，多少>0,多少<0'''
# def count_values(dictionary):
#     values = np.array(list(dictionary.values()))

#     greater_than_zero = np.sum(values > 0.0)
#     less_than_zero = np.sum(values < 0.0)
#     equal_to_zero = np.sum(values == 0.0)

#     return greater_than_zero, less_than_zero, equal_to_zero
def count_values(dictionary):
    values = torch.tensor(list(dictionary.values()))  # 将值转换为张量

    greater_than_zero = torch.sum(values > 0.0)
    less_than_zero = torch.sum(values < 0.0)
    equal_to_zero = torch.sum(values == 0.0)

    return greater_than_zero.item(), less_than_zero.item(), equal_to_zero.item()


'''不仅计算优劣节点的比例，看他们的优劣情况，具体距离0多远多近'''
def count_values2(dictionary):
    values = torch.tensor(list(dictionary.values()))  # 将值转换为张量

    greater_than_zero = values[values > 0.0]  # 保存大于0的所有值
    greater_than_zero_count = greater_than_zero.size(0)  # 大于0的值的数量
    less_than_zero = values[values < 0.0]  # 保存大于0的所有值
    less_than_zero_count = less_than_zero.size(0)  # 大于0的值的数量
   
    if greater_than_zero_count > 0:
        gmax_value = torch.max(greater_than_zero)  # 最大值
        gmin_value = torch.min(greater_than_zero)  # 最小值
        gav = torch.mean(greater_than_zero) 
        gerror = gmax_value - gmin_value  # 最大值和最小值之间的误差
    else:
        gmax_value = 7
        gmin_value = 7
        gav = 7
        gerror = 7

    if less_than_zero_count > 0:
        lmax_value = torch.max(less_than_zero)  # 最大值
        lmin_value = torch.min(less_than_zero)  # 最小值
        lav = torch.mean(less_than_zero) 
        lerror = lmax_value - lmin_value  # 最大值和最小值之间的误差
    else:
        lmax_value = 7
        lmin_value = 7
        lav = 7
        lerror = 7

    equal_to_zero = torch.sum(values == 0.0)

    return greater_than_zero_count, less_than_zero_count, equal_to_zero.item(),\
        gmax_value, gmin_value, gav, gerror,  lmax_value, lmin_value,lav, lerror 


def count_be_sa(dictionary):
    positive_values = []
    negative_values = []
    # 将大于0的值添加到positive_values列表中， 将小于0的值添加到negative_values列表中
    for value in dictionary.values():
        positive_values.append(value * (value > 0).float())
        negative_values.append(value * (value < 0).float())       

    # 在GPU上进行计算
    positive_sum = torch.sum(torch.stack(positive_values))
    negative_sum = torch.sum(torch.stack(negative_values))

    # 计算正数和负数之间的差并保留梯度
    '''不用绝对值'''
    difference = torch.abs(positive_sum + negative_sum)

    return difference









# 计算一段时间的每个节点的牺牲/受益状况，+-1？？dis_labels是+-1，dis_out是0-1之间
# 多采样牺牲的
def dynamic_cal5(dis_labels, dis_out, sample_map, district_nodes, district13_road_index, device):
    b,t,n,c = dis_labels.shape
    dis_out = dis_out.reshape(b,t,n,-1) # linear2后经过sigmoid，取值0-1表示受益/牺牲的状态
    # print("原来的输出：", dis_out)
    dis_out = dis_out - 0.5 # torch.where(dis_out > 0.5, dis_out - 0.5, 0.5 - dis_out)#.abs())
    # print("后来的输出：", dis_out)
    
    node_yl_dic = {} 
    node_yl_dic_disout = {} # 键为0-938，值为对应的dis_out的和 0.几

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
                
            # node_yl_dic[node_938] += torch.sum(dis_out_list[i][:,node].flatten()) # 应该是一个值，无数个0.几相加，针对digmoid输出
            #node_yl_dic[node_938] += torch.sum(dis_labels[:,:, node, :].flatten()) # 这段时间内的总体的受益/损失的加和，+-1
            node_yl_dic_disout[node_938] += torch.sum(dis_out[:,:, node, :].flatten()) # 某个节点这段时间的牺牲和受益总体情况，sum的是sigmoid后-0.5的值，可能大于或小于0
            
    # print("a:", len(list(node_yl_dic_disout.keys())))
    '''统计优劣的比例'''
    greater_than_zero, less_than_zero, equal_to_zero = count_values(node_yl_dic_disout)

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

    '''我们的目标是，越牺牲的越要采样'''
    # softmax算采样概率，不除以64和12，node_yl_dic原始506/-42，node_yl_dic_disout原始26/-2
    sum_node_list = list(district13_road_index.values()) # sum_node_list所有节点
    flat_node_list = [num for sublist in sum_node_list for num in sublist] # 展开2维list,所有节点938
    new_node_yl_dic, new_node_yl_dic_disout = dict(node_yl_dic), dict(node_yl_dic_disout) # node_yl_dic的修改不会影响new_node_yl_dic，它是独立的
    '''上一行用下一行替换，牺牲和受益取负数，我们的目标，多采样牺牲的'''
    # new_node_yl_dic_disout = {key: -value for key, value in node_yl_dic_disout.items()}
    for node_938 in flat_node_list: # 循环938个节点
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

    return differences_sum_dout, values_node, greater_than_zero, less_than_zero, equal_to_zero # 基于0.x的loss和list节点采样概率
    # sorted_new_node_yl_dic, sorted_new_node_yl_dic_disout, differences_sum_dlabel, differences_sum_dout, softmax_lvalues, softmax_dvalues
    # +-1字典，0.x字典，Dloss值，Dloss值, softmax的list, softmax的list2


# 多采样受益节点
def dynamic_cal5_2(dis_labels, dis_out, sample_map, district_nodes, district13_road_index, device):
    b,t,n,c = dis_labels.shape
    dis_out = dis_out.reshape(b,t,n,-1) # linear2后经过sigmoid，取值0-1表示受益/牺牲的状态
    # print("原来的输出：", dis_out)
    dis_out = dis_out - 0.5 # torch.where(dis_out > 0.5, dis_out - 0.5, 0.5 - dis_out)#.abs())
    # print("后来的输出：", dis_out)
    
    node_yl_dic = {} 
    node_yl_dic_disout = {} # 键为0-938，值为对应的dis_out的和 0.几
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
                
            # node_yl_dic[node_938] += torch.sum(dis_out_list[i][:,node].flatten()) # 应该是一个值，无数个0.几相加，针对digmoid输出
            #node_yl_dic[node_938] += torch.sum(dis_labels[:,:, node, :].flatten()) # 这段时间内的总体的受益/损失的加和，+-1
            node_yl_dic_disout[node_938] += torch.sum(dis_out[:,:, node, :].flatten()) # 某个节点这段时间的牺牲和受益总体情况，sum的是sigmoid后-0.5的值，可能大于或小于0

    # print("a:", len(list(node_yl_dic_disout.keys())))
    '''统计优劣的比例'''
    greater_than_zero, less_than_zero, equal_to_zero = count_values(node_yl_dic_disout)

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

    '''我们的目标是，越牺牲的越要采样'''
    # softmax算采样概率，不除以64和12，node_yl_dic原始506/-42，node_yl_dic_disout原始26/-2
    sum_node_list = list(district13_road_index.values()) # sum_node_list所有节点
    flat_node_list = [num for sublist in sum_node_list for num in sublist] # 展开2维list,所有节点938
    #new_node_yl_dic, new_node_yl_dic_disout = dict(node_yl_dic), dict(node_yl_dic_disout) # node_yl_dic的修改不会影响new_node_yl_dic，它是独立的
    '''上一行用下一行替换，牺牲和受益取负数，我们的目标，多采样牺牲的'''
    new_node_yl_dic_disout = {key: -value for key, value in node_yl_dic_disout.items()}
    for node_938 in flat_node_list: # 循环938个节点
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

    return differences_sum_dout, values_node, greater_than_zero, less_than_zero, equal_to_zero # 基于0.x的loss和list节点采样概率
    # sorted_new_node_yl_dic, sorted_new_node_yl_dic_disout, differences_sum_dlabel, differences_sum_dout, softmax_lvalues, softmax_dvalues
    # +-1字典，0.x字典，Dloss值，Dloss值, softmax的list, softmax的list2



# 不是基于鉴别器输出，而是ground_truth和pred
def dynamic_cal6(yl_label2, dis_out, sample_map, district_nodes, district13_road_index, device):
    b,t,n,c = yl_label2.shape # 全是+-1，根据ground_truth
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
                node_yl_dic[node_938] = 0
                node_yl_dic_disout[node_938] = 0
                
            node_yl_dic[node_938] += torch.sum(yl_label2[:,:, node, :].flatten())
            node_yl_dic_disout[node_938] += torch.sum(dis_out[:,:, node, :].flatten())

    # 计算动态损失：只基于采样节点
    differences_dlabel = []
    node_keys = list(node_yl_dic.keys()) # 这段时间出现过的采样节点，0-938
    num_nodes = len(node_keys) # 可能len为500(>450) # 一个batch内就是450
    # 计算节点两两之间的差值，只考虑了采样出来的节点
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            diff_dout = torch.abs(node_yl_dic[node_keys[i]] - node_yl_dic[node_keys[j]]) # 两个值相减
            differences_dlabel.append(diff_dout)
    differences_sum_dout = torch.mean(torch.tensor(differences_dlabel)) # 基于dis_out, 0.x计算出的动态损失


    # softmax算采样概率，不除以64和12，node_yl_dic原始506/-42，node_yl_dic_disout原始26/-2
    sum_node_list = list(district13_road_index.values()) # sum_node_list所有节点
    flat_node_list = [num for sublist in sum_node_list for num in sublist] # 展开2维list,所有节点938
    new_node_yl_dic_disout = dict(node_yl_dic_disout) # node_yl_dic的修改不会影响new_node_yl_dic，它是独立的
    for node_938 in flat_node_list:
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

    return differences_sum_dout, values_node









# 10月讨论新加，先不考虑动静态损失对sample的影响，将采样看作优化问题
'''
优化目标
1. 缓解数据不平衡  2. 确保采样的多样性

已知：香港数据938个节点，采样450个节点，对于baseline采样不变
1. sample_list采样列表，长度为450的list，里面元素取值0-937
2. sample_map字典，键为0-449，值为0-937
3. sample_dict字典，键为0-12（13个区域），值为list，list中元素对于该区域节点下标0-937
4. district_nodes字典，键为0-12，值为list，每个区域节点下标0-449

目标：根据已知条件和优化目标求出最适合的采样方法！
#### 增加总体受益/牺牲的平衡，所有受益 尽可能 = 所有牺牲
'''
# 相较与dynamic_cal5，多了node_yl_dic_disout_global字典，用于贯穿整个训练过程
def dynamic_cal5_global(dis_labels, dis_out, sample_map, district_nodes): # , district13_road_index, node_count_global):
    b,t,n,c = dis_labels.shape
    dis_out = dis_out.reshape(b,t,n,-1) # linear2后经过sigmoid，取值0-1表示受益/牺牲的状态
    # print("原来的输出：", dis_out)
    dis_out = dis_out - 0.5 # torch.where(dis_out > 0.5, dis_out - 0.5, 0.5 - dis_out)#.abs())
    # print("后来的输出：", dis_out)
    # print("1. dis_out requires_grad:", dis_out.requires_grad)
    node_yl_dic = {} 
    node_yl_dic_disout = {} # 键为0-938，值为对应的dis_out的和 0.几

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
            '''采样的时候算过了'''
            # if node_938 not in node_count_global: 
            #     node_count_global[node_938] = 0
                
            # node_yl_dic[node_938] += torch.sum(dis_out_list[i][:,node].flatten()) # 应该是一个值，无数个0.几相加，针对digmoid输出
            #node_yl_dic[node_938] += torch.sum(dis_labels[:,:, node, :].flatten()) # 这段时间内的总体的受益/损失的加和，+-1
            '''原来dis_out,取值0-1，dis_label取值1/-1'''
            node_yl_dic_disout[node_938] += torch.sum(dis_out[:,:, node, :].flatten()) # 某个节点这段时间的牺牲和受益总体情况，sum的是sigmoid后-0.5的值，可能大于或小于0
            # print("2. node_yl_dic_disout requires_grad:", node_yl_dic_disout[node_938].requires_grad) # T
            # node_count_global[node_938] += 1 # 计数，采样过几次

    '''约束受益和牺牲，不能大家都牺牲'''
    yl_difference = count_be_sa(node_yl_dic_disout) 

    '''统计优劣的比例'''
    # greater_than_zero, less_than_zero, equal_to_zero = count_values2(node_yl_dic_disout)
    greater_than_zero, less_than_zero, equal_to_zero,\
        gmax_value, gmin_value, gav, gerror,  lmax_value, lmin_value,lav, lerror = count_values2(node_yl_dic_disout)

       
    # 计算动态损失：只基于采样节点
    differences_dlabel, differences_dout = [],[]
    node_keys = list(node_yl_dic_disout.keys()) # 这段时间出现过的采样节点，0-938
    num_nodes = len(node_keys) # 可能len为500(>450) # 一个batch内就是450
    # 计算节点两两之间的差值，只考虑了采样出来的节点
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            #diff_dlabel = torch.abs(node_yl_dic[node_keys[i]] - node_yl_dic[node_keys[j]]) # 两个值相减
            diff_dout = torch.abs(node_yl_dic_disout[node_keys[i]] - node_yl_dic_disout[node_keys[j]]) # 两个值相减
            # print("1. diff_dout requires_grad:", diff_dout.requires_grad) # T
            #differences_dlabel.append(diff_dlabel)
            differences_dout.append(diff_dout)
            
    #differences_sum_dlabel = torch.mean(torch.tensor(differences_dlabel)) # 基于+-1计算出的动态损失
    differences_sum_dout = torch.mean(torch.stack(differences_dout), dim=0) # torch.mean(torch.tensor(differences_dout))没有梯度！！ # 基于dis_out, 0.x计算出的动态损失
    # print("1. differences_sum_dout requires_grad:", differences_sum_dout.requires_grad) # T
    return differences_sum_dout, yl_difference, greater_than_zero, less_than_zero, equal_to_zero, \
         gmax_value, gmin_value, gav, gerror,  lmax_value, lmin_value,lav, lerror # , node_count_global


def dynamic_cal5_global_no5(dis_labels, dis_out, sample_map, district_nodes): # , district13_road_index, node_count_global):
    b,t,n,c = dis_labels.shape
    dis_out = dis_out.reshape(b,t,n,-1) # linear2后经过sigmoid，取值0-1表示受益/牺牲的状态
    # print("原来的输出：", dis_out)
    dis_out = dis_out - 0.5 # torch.where(dis_out > 0.5, dis_out - 0.5, 0.5 - dis_out)#.abs())
    # print("后来的输出：", dis_out)
    # print("1. dis_out requires_grad:", dis_out.requires_grad)
    node_yl_dic = {} 
    node_yl_dic_disout = {} # 键为0-938，值为对应的dis_out的和 0.几
    
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
            '''采样的时候算过了'''
            # if node_938 not in node_count_global: 
            #     node_count_global[node_938] = 0
                
            # node_yl_dic[node_938] += torch.sum(dis_out_list[i][:,node].flatten()) # 应该是一个值，无数个0.几相加，针对digmoid输出
            #node_yl_dic[node_938] += torch.sum(dis_labels[:,:, node, :].flatten()) # 这段时间内的总体的受益/损失的加和，+-1
            '''原来dis_out,取值0-1，dis_label取值1/-1'''
            node_yl_dic_disout[node_938] += torch.sum(dis_out[:,:, node, :].flatten()) # 某个节点这段时间的牺牲和受益总体情况，sum的是sigmoid后-0.5的值，可能大于或小于0
            # print("2. node_yl_dic_disout requires_grad:", node_yl_dic_disout[node_938].requires_grad) # T
            # node_count_global[node_938] += 1 # 计数，采样过几次

    # '''约束受益和牺牲，不能大家都牺牲'''
    # yl_difference = count_be_sa(node_yl_dic_disout) 

    '''统计优劣的比例'''
    # greater_than_zero, less_than_zero, equal_to_zero = count_values2(node_yl_dic_disout)
    greater_than_zero, less_than_zero, equal_to_zero,\
        gmax_value, gmin_value, gav, gerror,  lmax_value, lmin_value,lav, lerror = count_values2(node_yl_dic_disout)

       
    # 计算动态损失：只基于采样节点
    differences_dlabel, differences_dout = [],[]
    node_keys = list(node_yl_dic_disout.keys()) # 这段时间出现过的采样节点，0-938
    num_nodes = len(node_keys) # 可能len为500(>450) # 一个batch内就是450
    # 计算节点两两之间的差值，只考虑了采样出来的节点
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            #diff_dlabel = torch.abs(node_yl_dic[node_keys[i]] - node_yl_dic[node_keys[j]]) # 两个值相减
            diff_dout = torch.abs(node_yl_dic_disout[node_keys[i]] - node_yl_dic_disout[node_keys[j]]) # 两个值相减
            # print("1. diff_dout requires_grad:", diff_dout.requires_grad) # T
            #differences_dlabel.append(diff_dlabel)
            differences_dout.append(diff_dout)
            
    #differences_sum_dlabel = torch.mean(torch.tensor(differences_dlabel)) # 基于+-1计算出的动态损失
    differences_sum_dout = torch.mean(torch.stack(differences_dout), dim=0) # torch.mean(torch.tensor(differences_dout))没有梯度！！ # 基于dis_out, 0.x计算出的动态损失
    # print("1. differences_sum_dout requires_grad:", differences_sum_dout.requires_grad) # T
    return differences_sum_dout, greater_than_zero, less_than_zero, equal_to_zero, \
         gmax_value, gmin_value, gav, gerror,  lmax_value, lmin_value,lav, lerror # , node_count_global


# 提高效率，不计算具体数值
def dynamic_cal5_global_no5_2(dis_labels, dis_out, sample_map, district_nodes): # , district13_road_index, node_count_global):
    b,t,n,c = dis_labels.shape
    dis_out = dis_out.reshape(b,t,n,-1) # linear2后经过sigmoid，取值0-1表示受益/牺牲的状态
    # print("原来的输出：", dis_out)
    dis_out = dis_out - 0.5 # torch.where(dis_out > 0.5, dis_out - 0.5, 0.5 - dis_out)#.abs())
    # print("后来的输出：", dis_out)
    # print("1. dis_out requires_grad:", dis_out.requires_grad)
    node_yl_dic = {} 
    node_yl_dic_disout = {} # 键为0-938，值为对应的dis_out的和 0.几
    
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
            '''采样的时候算过了'''
            # if node_938 not in node_count_global: 
            #     node_count_global[node_938] = 0
                
            # node_yl_dic[node_938] += torch.sum(dis_out_list[i][:,node].flatten()) # 应该是一个值，无数个0.几相加，针对digmoid输出
            #node_yl_dic[node_938] += torch.sum(dis_labels[:,:, node, :].flatten()) # 这段时间内的总体的受益/损失的加和，+-1
            '''原来dis_out,取值0-1，dis_label取值1/-1'''
            node_yl_dic_disout[node_938] += torch.sum(dis_out[:,:, node, :].flatten()) # 某个节点这段时间的牺牲和受益总体情况，sum的是sigmoid后-0.5的值，可能大于或小于0
            # print("2. node_yl_dic_disout requires_grad:", node_yl_dic_disout[node_938].requires_grad) # T
            # node_count_global[node_938] += 1 # 计数，采样过几次

    # '''约束受益和牺牲，不能大家都牺牲'''
    # yl_difference = count_be_sa(node_yl_dic_disout) 

    '''统计优劣的比例'''
    # greater_than_zero, less_than_zero, equal_to_zero = count_values2(node_yl_dic_disout)
    greater_than_zero, less_than_zero, equal_to_zero = count_values(node_yl_dic_disout) # 总体情况。每个节点在一段时间内的状态

       
    # 计算动态损失：只基于采样节点
    differences_dlabel, differences_dout = [],[]
    node_keys = list(node_yl_dic_disout.keys()) # 这段时间出现过的采样节点，0-938
    num_nodes = len(node_keys) # 可能len为500(>450) # 一个batch内就是450
    # 计算节点两两之间的差值，只考虑了采样出来的节点
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            #diff_dlabel = torch.abs(node_yl_dic[node_keys[i]] - node_yl_dic[node_keys[j]]) # 两个值相减
            diff_dout = torch.abs(node_yl_dic_disout[node_keys[i]] - node_yl_dic_disout[node_keys[j]]) # 两个值相减
            # print("1. diff_dout requires_grad:", diff_dout.requires_grad) # T
            #differences_dlabel.append(diff_dlabel)
            differences_dout.append(diff_dout)
            
    #differences_sum_dlabel = torch.mean(torch.tensor(differences_dlabel)) # 基于+-1计算出的动态损失
    differences_sum_dout = torch.mean(torch.stack(differences_dout), dim=0) # torch.mean(torch.tensor(differences_dout))没有梯度！！ # 基于dis_out, 0.x计算出的动态损失
    # print("1. differences_sum_dout requires_grad:", differences_sum_dout.requires_grad) # T
    return differences_sum_dout, greater_than_zero, less_than_zero, equal_to_zero


# 计算的loss4是根据0的得到的，要求都距离0比较近
def dynamic_cal5_global_no5_3(dis_labels, dis_out, sample_map, district_nodes, node_yl_dic_disout): # , district13_road_index, node_count_global):
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
            '''采样的时候算过了'''
            node_yl_dic_disout[node_938] += torch.sum(dis_out[:,:, node, :].flatten()) # 某个节点这段时间的牺牲和受益总体情况，sum的是sigmoid后-0.5的值，可能大于或小于0

    '''统计优劣的比例'''
    greater_than_zero, less_than_zero, equal_to_zero = count_values(node_yl_dic_disout) # 总体情况。每个节点在一段时间内的状态
       
    # 计算动态损失：只基于采样节点
    differences_dlabel, differences_dout = [],[]
    node_keys = list(node_yl_dic_disout.keys()) # 这段时间出现过的采样节点，0-938
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

# xhbl专用，算了node_global
def dynamic_cal5_global_xhbl(dis_labels, dis_out, sample_map, district_nodes, district13_road_index, node_count_global):
    b,t,n,c = dis_labels.shape
    dis_out = dis_out.reshape(b,t,n,-1) # linear2后经过sigmoid，取值0-1表示受益/牺牲的状态
    
    dis_out = dis_out - 0.5 # torch.where(dis_out > 0.5, dis_out - 0.5, 0.5 - dis_out)#.abs())
    
    node_yl_dic = {} 
    node_yl_dic_disout = {} # 键为0-938，值为对应的dis_out的和 0.几
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
            '''采样的时候算过了'''
            if node_938 not in node_count_global: 
                node_count_global[node_938] = 0
                
            # node_yl_dic[node_938] += torch.sum(dis_out_list[i][:,node].flatten()) # 应该是一个值，无数个0.几相加，针对digmoid输出
            #node_yl_dic[node_938] += torch.sum(dis_labels[:,:, node, :].flatten()) # 这段时间内的总体的受益/损失的加和，+-1
            '''原来dis_out,取值0-1，dis_label取值1/-1'''
            node_yl_dic_disout[node_938] += torch.sum(dis_out[:,:, node, :].flatten()) # 某个节点这段时间的牺牲和受益总体情况，sum的是sigmoid后-0.5的值，可能大于或小于0
            # print("2. node_yl_dic_disout requires_grad:", node_yl_dic_disout[node_938].requires_grad) # T
            node_count_global[node_938] += 1 # 计数，采样过几次
              
    # 计算动态损失：只基于采样节点
    differences_dlabel, differences_dout = [],[]
    node_keys = list(node_yl_dic_disout.keys()) # 这段时间出现过的采样节点，0-938
    num_nodes = len(node_keys) # 可能len为500(>450) # 一个batch内就是450
    # 计算节点两两之间的差值，只考虑了采样出来的节点
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            #diff_dlabel = torch.abs(node_yl_dic[node_keys[i]] - node_yl_dic[node_keys[j]]) # 两个值相减
            diff_dout = torch.abs(node_yl_dic_disout[node_keys[i]] - node_yl_dic_disout[node_keys[j]]) # 两个值相减
            # print("1. diff_dout requires_grad:", diff_dout.requires_grad) # T
            #differences_dlabel.append(diff_dlabel)
            differences_dout.append(diff_dout)
            
    #differences_sum_dlabel = torch.mean(torch.tensor(differences_dlabel)) # 基于+-1计算出的动态损失
    differences_sum_dout = torch.mean(torch.stack(differences_dout), dim=0) # torch.mean(torch.tensor(differences_dout))没有梯度！！ # 基于dis_out, 0.x计算出的动态损失
    # print("1. differences_sum_dout requires_grad:", differences_sum_dout.requires_grad) # T
    return differences_sum_dout, node_count_global

'''对dis_out处理改变！让其变成-1/1'''
def dynamic_cal5_global2(dis_labels, dis_out, sample_map, district_nodes, district13_road_index, node_count_global):
    b,t,n,c = dis_labels.shape
    dis_out = dis_out.reshape(b,t,n,-1) # linear2后经过sigmoid，取值0-1表示受益/牺牲的状态
    # dis_out = dis_out - 0.5 
    # print("1. dis_out requires_grad:", dis_out.requires_grad)
    dis_out = torch.where(dis_out > 0.5, torch.ones_like(dis_out, requires_grad=True), torch.full_like(dis_out, -1, requires_grad=True))        
    # print("2. dis_out requires_grad:", dis_out.requires_grad)
    node_yl_dic = {} 
    node_yl_dic_disout = {} # 键为0-938，值为对应的dis_out的和 0.几
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
                
            # node_yl_dic[node_938] += torch.sum(dis_out_list[i][:,node].flatten()) # 应该是一个值，无数个0.几相加，针对digmoid输出
            #node_yl_dic[node_938] += torch.sum(dis_labels[:,:, node, :].flatten()) # 这段时间内的总体的受益/损失的加和，+-1
            '''原来dis_out,取值0-1，dis_label取值1/-1'''
            node_yl_dic_disout[node_938] += torch.sum(dis_out[:,:, node, :].flatten()) # 某个节点这段时间的牺牲和受益总体情况，sum的是sigmoid后-0.5的值，可能大于或小于0
            # print("2. node_yl_dic_disout requires_grad:", node_yl_dic_disout[node_938].requires_grad) # T
            node_count_global[node_938] += 1 # 计数，采样过几次
              
    # 计算动态损失：只基于采样节点
    differences_dlabel, differences_dout = [],[]
    node_keys = list(node_yl_dic_disout.keys()) # 这段时间出现过的采样节点，0-938
    num_nodes = len(node_keys) # 可能len为500(>450) # 一个batch内就是450
    # 计算节点两两之间的差值，只考虑了采样出来的节点
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            #diff_dlabel = torch.abs(node_yl_dic[node_keys[i]] - node_yl_dic[node_keys[j]]) # 两个值相减
            diff_dout = torch.abs(node_yl_dic_disout[node_keys[i]] - node_yl_dic_disout[node_keys[j]]) # 两个值相减
            # print("1. diff_dout requires_grad:", diff_dout.requires_grad) # T
            #differences_dlabel.append(diff_dlabel)
            differences_dout.append(diff_dout)
            
    #differences_sum_dlabel = torch.mean(torch.tensor(differences_dlabel)) # 基于+-1计算出的动态损失
    differences_sum_dout = torch.mean(torch.stack(differences_dout), dim=0) # torch.mean(torch.tensor(differences_dout))没有梯度！！ # 基于dis_out, 0.x计算出的动态损失
    # print("1. differences_sum_dout requires_grad:", differences_sum_dout.requires_grad) # T

    return differences_sum_dout, node_count_global

'''基于dis_out来梯度下降（还是-0.5），返回一个follow dis_out,计算的-1/1的值'''
def dynamic_cal5_global3(dis_labels, dis_out, sample_map, district_nodes, district13_road_index, node_count_global):
    b,t,n,c = dis_labels.shape
    dis_out = dis_out.reshape(b,t,n,-1) # linear2后经过sigmoid，取值0-1表示受益/牺牲的状态
    # print("原来的输出：", dis_out)
    dis_out = dis_out - 0.5 # torch.where(dis_out > 0.5, dis_out - 0.5, 0.5 - dis_out)#.abs())
    # print("后来的输出：", dis_out)
    # print("1. dis_out requires_grad:", dis_out.requires_grad)
    dis_labels_new = torch.where(dis_out > 0, torch.ones_like(dis_out), torch.full_like(dis_out, -1))
    node_yl_dic = {} 
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