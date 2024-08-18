
# coding: utf-8

'''
嵌套了香港数据集试验
这段代码有以下几个特点：

目标1的权重根据历史采样结果动态计算。具体来说，第一轮采样时，设目标1的权重为1；之后每轮采样结束后，根据上一轮采样结果的目标1和目标2的得分计算权重，采用指数衰减算法，使得历史结果对当前结果的影响越来越小。
每轮先选取每个区域中的一个节点，保证每个区域至少有一个节点被选择。
剩余的节点采用贪心算法选择，对于每个节点，计算它在目标1和目标2下的得分，然后按总得分从大到小排序，选取得分最高的节点加入采样结果中。
目标2的计算借助了历史采样结果。具体来说，对于当前要采样的节点，遍历历史采样结果，计算它在历史采样结果中的出现次数，并将其除以历史采样结果的长度得到平均出现次数，作为该节点的目标2得分。
希望这次回答能够满足您的要求，如果还有问题，请随时提问！

贪心：先确定区域，再确定具体节点
'''

# from src.engines.sample import *
import numpy as np
import os
import json
import random
import copy
import torch

# 先区域后节点采样
def optimize_selection(nodes_by_region, sample_num, node_count_global):
    '''
    nodes_by_region：所有节点分区情况，937个节点
    total_nodes：所有节点的列表[0,1,2,...,937]
    sample_num：采样总数450
    node_count_global：字典统计每个节点被选择的次数，键0-937，值为对应节点的采样次数
    '''
    selected_nodes = []
    selected_counts = {} # 区域字典，键为区域，值为该区域的采样节点计数
    '''copy的新字典不对原始的区域节点列表产生影响。'''
    nodes_by_region_copy =  {r: nodes_by_region[r][:] for r in list(nodes_by_region.keys())} # 创建一个区域节点的副本，以便在贪心算法中进行节点选择时
    mean_num = int(sample_num/len(list(nodes_by_region.keys()))) # 平均一个区域要采样的数目. 34
   
    # 按照目标1选择节点，每个区域至少选一个
    for region in list(nodes_by_region.keys()): # 循环每个区域
        if region not in selected_counts:
            selected_counts[region] = 0
        selected_counts[region] += 1 # 区域选择计数+1
        

        if len(nodes_by_region[region]) > 0: # 如果该区域有节点
            node = random.choice(nodes_by_region[region]) # 随机选一个
            selected_nodes.append(node) # 加入选择列表
            nodes_by_region_copy[region].remove(node) # 在copy的区域字典中，删除采样过的字典
            # if region not in selected_counts:
            #     selected_counts[region] = 0
            # selected_counts[region] += 1 # 区域选择计数+1
            # nodes_by_region_copy[region].remove(node) # 在copy的区域字典中，删除采样过的字典

    n_nodes_left = sample_num - len(selected_nodes) # 每个区域选择一个后，剩下的可选择次数
    
    i =1
    # 使用贪心算法选择剩余的节点
    while n_nodes_left > 0:
        max_score = float('-inf')
        selected_region = None
        selected_node = None
        score1_list, region_list = [], []
        score2_list = []
        
        # 先区域，再节点
        for region, nodes in nodes_by_region_copy.items():
            '''选区域'''
            if len(nodes) > 0: # 确保区域还有节点，没有节点的话，socre1持续为负数但无法取值，会陷入死循环
                score1 = selected_counts[region] + 1 - mean_num # 越小越要选, 负数sigmoid后靠近0
                score1_list.append(score1)
                region_list.append(region)
            else:
                continue
        region_min_index = torch.argmin(torch.tensor(score1_list))
        selected_region = region_list[region_min_index.item()]

        if selected_region is None: # 某一区域的节点选完了！双重保险
            continue

        '''选节点,确定区域后'''
        nodes_list = nodes_by_region_copy[selected_region]
        for node in nodes_list:
            score2 = node_count_global.get(node, 0)  # 同一区域的节点，越小越要选
            score2_list.append(score2)
        node_min_index = torch.argmin(torch.tensor(score2_list))
        selected_node = nodes_list[node_min_index.item()]

        
        selected_nodes.append(selected_node) # 添加一个节点
        '''更新了一个节点，更新selected_counts,键为区域，值为对应区域的采样数目'''
        if selected_region not in selected_counts:
            selected_counts[selected_region] = 0
        selected_counts[selected_region] += 1

        nodes_by_region_copy[selected_region].remove(selected_node) # 删除某区域中选择过的节点

        '''更新了一个节点，更新node_count_global'''
        if selected_node not in node_count_global:
            node_count_global[selected_node] = 0
        node_count_global[selected_node] += 1
        
        n_nodes_left -= 1
        i +=1

    return selected_nodes, node_count_global


# 区域和节点同级别采样，没有先区域后节点
def optimize_selection_equal(nodes_by_region, sample_num, node_count_global, device):
    '''
    nodes_by_region：所有节点分区情况，937个节点
    total_nodes：所有节点的列表[0,1,2,...,937]
    sample_num：采样总数450
    node_count_global：字典统计每个节点被选择的次数，键0-937，值为对应节点的采样次数
    '''
    selected_nodes = []
    selected_counts = {} # 区域字典，键为区域，值为该区域的采样节点计数
    '''copy的新字典不对原始的区域节点列表产生影响。'''
    nodes_by_region_copy =  {r: nodes_by_region[r][:] for r in list(nodes_by_region.keys())} # 创建一个区域节点的副本，以便在贪心算法中进行节点选择时
    mean_num = int(sample_num/len(list(nodes_by_region.keys()))) # 平均一个区域要采样的数目. 34
   
    # 按照目标1选择节点，每个区域至少选一个
    for region in list(nodes_by_region.keys()): # 循环每个区域
        if region not in selected_counts:
            selected_counts[region] = 0
        selected_counts[region] += 1 # 区域选择计数+1
        

        if len(nodes_by_region[region]) > 0: # 如果该区域有节点
            node = random.choice(nodes_by_region[region]) # 随机选一个
            selected_nodes.append(node) # 加入选择列表
            nodes_by_region_copy[region].remove(node) # 在copy的区域字典中，删除采样过的字典
            
    n_nodes_left = sample_num - len(selected_nodes) # 每个区域选择一个后，剩下的可选择次数
    
    i =1
    # 使用贪心算法选择剩余的节点
    while n_nodes_left > 0:
        max_score = float('-inf')
        selected_region = None
        selected_node = None
      
        score_list = []
        
        # 区域节点并列
        for region, nodes in nodes_by_region_copy.items():
            if len(nodes) > 0:
                for node in nodes:
                    score1 = selected_counts[region] + 1 - mean_num # 越小越要选, 负数sigmoid后靠近0
                    score2 = node_count_global.get(node, 0)  # 越小越要选
                    score_list.append((score1, score2, node, region))


        score1_list = list(zip(*score_list))[0]
        score2_list = list(zip(*score_list))[1]
        nodes_list = list(zip(*score_list))[2]
        region_list = list(zip(*score_list))[3]
        result1 = process_tensor(torch.tensor(score1_list), device) # 返回归一化的区域参数，每个区域距离均值的距离，越小越要采样
        result2 = process_tensor(torch.tensor(score2_list), device) # 返回归一化的区域参数，每个区域距离均值的距离，越小越要采样
        
        score = result1 * result2
        region_min_index = torch.argmin(score)
        selected_region = region_list[region_min_index.item()]
        selected_node = nodes_list[region_min_index.item()]

        if selected_region is None: # 某一区域的节点选完了！双重保险
            continue
        
        selected_nodes.append(selected_node) # 添加一个节点
        '''更新了一个节点，更新selected_counts,键为区域，值为对应区域的采样数目'''
        if selected_region not in selected_counts:
            selected_counts[selected_region] = 0
        selected_counts[selected_region] += 1

        nodes_by_region_copy[selected_region].remove(selected_node) # 删除某区域中选择过的节点

        '''更新了一个节点，更新node_count_global'''
        if selected_node not in node_count_global:
            node_count_global[selected_node] = 0
        node_count_global[selected_node] += 1
        
        n_nodes_left -= 1
        i +=1

    return selected_nodes, node_count_global




# 判断tensor中所有元素是否一致, 一致直接sigmoid，不一致先归一化再sigmoid
def process_tensor(tensor, device):
    tensor = tensor.to(device)
    if torch.all(tensor == tensor[0]):
        # 所有元素一致，直接进行 sigmoid 操作
        tensor.fill_(0.0)
        result = torch.sigmoid(tensor)
    else:
        # 元素不一致，先进行归一化，再进行 sigmoid 操作
        normalized_tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        result = torch.sigmoid(normalized_tensor)

    return result


def optimize_selection_withfair(values_region, values_node, nodes_by_region, sample_num, node_count_global, device):
    # print("1:values_region", values_region, type(values_region), values_region.shape) # [13]
    # print("2:values_node", values_node, type(values_node), values_node.shape) # [938]
   
    '''
    values_region: 静态损失来的反馈，tensor，13个区域，每个区域的误差情况(mape), 误差越小概率越小。多采样误差大的？
    values_node: 动态损失来的反馈，tensor，938个节点。每个节点受益/牺牲情况，牺牲的话概率就小，受益的话概率就大。多采样牺牲的？
    原：多采误差大的区域，多采受益的节点
    现在：多采误差大的区域，多采受益的节点
    nodes_by_region：所有节点分区情况，937个节点
    total_nodes：所有节点的列表[0,1,2,...,937]
    sample_num：采样总数450
    node_count_global：字典统计每个节点被选择的次数，键0-937，值为对应节点的采样次数
    '''
    selected_nodes = []
    selected_counts = {} # 区域字典，键为区域，值为该区域的采样节点计数
    '''copy的新字典不对原始的区域节点列表产生影响。nodes_by_region是原始字典0-12：938'''
    nodes_by_region_copy =  {r: nodes_by_region[r][:] for r in list(nodes_by_region.keys())} # 创建一个区域节点的副本，以便在贪心算法中进行节点选择时
    mean_num = int(sample_num/len(list(nodes_by_region.keys()))) # 平均一个区域要采样的数目. 34
   
    # 按照目标1选择节点，每个区域至少选一个
    for region in list(nodes_by_region.keys()): # 循环每个区域
        if region not in selected_counts:
            selected_counts[region] = 0
        selected_counts[region] += 1 # 区域选择计数+1
        

        if len(nodes_by_region[region]) > 0: # 如果该区域有节点
            node = random.choice(nodes_by_region[region]) # 随机选一个
            selected_nodes.append(node) # 加入选择列表
            nodes_by_region_copy[region].remove(node) # 在copy的区域字典中，删除采样过的字典
            # if region not in selected_counts:
            #     selected_counts[region] = 0
            # selected_counts[region] += 1 # 区域选择计数+1
            # nodes_by_region_copy[region].remove(node) # 在copy的区域字典中，删除采样过的字典

    n_nodes_left = sample_num - len(selected_nodes) # 每个区域选择一个后，剩下的可选择次数
    
    i =1
    # 使用贪心算法选择剩余的节点
    while n_nodes_left > 0:
        max_score = float('-inf')
        selected_region = None
        selected_node = None
        score1_list, score1_list2, region_list = [], [], []
        score2_list, score2_list2= [],[]
        
        # 先区域，再节点
        for region, nodes in nodes_by_region_copy.items():
            '''选区域'''
            if len(nodes) > 0: # 确保区域还有节点，没有节点的话，socre1持续为负数但无法取值，会陷入死循环
                score1 = selected_counts[region] + 1 - mean_num # 越小越要选, 负数sigmoid后靠近0
                score1_list.append(score1)
                region_list.append(region)
                score1_list2.append(values_region[int(region)]) # str转int
            else:
                continue
        
        '''先选择一个区域'''
        result1 = process_tensor(torch.tensor(score1_list), device) # 返回归一化的区域参数，每个区域距离均值的距离，越小越要采样
        result2 = torch.stack(score1_list2) # 保持list中每个元素的梯度，有梯度的
       # print("a",result1.device, result2.device, score1_list2[0].device, values_region.device) 
        result12 = result1*result2
        # print(result12, result12.requires_grad) # [13] True
        
        region_min_index = torch.argmin(result12)
        selected_region = region_list[region_min_index.item()]

        if selected_region is None: # 某一区域的节点选完了！双重保险
            continue

        '''选节点,确定区域后'''
        nodes_list = nodes_by_region_copy[selected_region]
        for node in nodes_list:
            score2 = node_count_global.get(node, 0)  # 同一区域的节点，越小越要选
            score2_list.append(score2)
            score2_list2.append(values_node[node]) # node取值0-938
        
        result3 = process_tensor(torch.tensor(score2_list), device) # 返回归一化的区域参数，每个区域距离均值的距离，越小越要采样
        result4 = torch.stack(score2_list2)
        result34 = result3*result4
        # print(result3, result3.requires_grad, result3.device)
        # print(result4, result4.requires_grad, result4.device)
        # # print(result34.requires_grad)
        # pirnt()

        node_min_index = torch.argmin(result34)
        selected_node = nodes_list[node_min_index.item()]

        
        selected_nodes.append(selected_node) # 添加一个节点
        '''更新了一个节点，更新selected_counts,键为区域，值为对应区域的采样数目'''
        if selected_region not in selected_counts:
            selected_counts[selected_region] = 0
        selected_counts[selected_region] += 1

        nodes_by_region_copy[selected_region].remove(selected_node) # 删除某区域中选择过的节点

        '''更新了一个节点，更新node_count_global'''
        if selected_node not in node_count_global:
            node_count_global[selected_node] = 0
        node_count_global[selected_node] += 1
        
        n_nodes_left -= 1
        i +=1

    return selected_nodes, node_count_global





'''把新的采样结果，sample_list添加到node_count_global字典中'''
def calcute_global_dic(node_count_global, sample_list):

    for node in sample_list:
        if node not in node_count_global:
            node_count_global[node] = 0
        node_count_global[node] += 1
    return node_count_global


# def get_dataset_info(dataset):
#     # base_dir = os.getcwd() + '/data/' # 不适合服务器
#     base_dir ='/home/data/xjn/23largest_baseline/LargeST/data/'
#     d = {
#          'CA': [base_dir+'ca', base_dir+'ca/ca_rn_adj.npy', 8600],
#          'GLA': [base_dir+'gla', base_dir+'gla/gla_rn_adj.npy', 3834],
#          'GBA': [base_dir+'gba', base_dir+'gba/gba_rn_adj.npy', 2352],
#         #  'SD': [base_dir+'sd', base_dir+'sd/sd_rn_adj.npy', 716],
#          'SD': [base_dir+'sd', base_dir+'sd/2019/adj_initial350_all1.npy', 716], # 全1/0的矩阵
#          'HK': [base_dir+'hk', base_dir+'hk/202010/adj_initial450_all1.npy', 938], 
#         }
#     assert dataset in d.keys()
#     return d[dataset]





# '''当前文件运行测试'''
# def main():

#     data_path, adj_path, node_num = get_dataset_info("HK")
#     # 初始化的分布，初始采样，baseline不变，Fairness每个batch改变！
#     ptr1 = np.load(os.path.join(data_path, "202010", 'his_initial450.npz'), allow_pickle=True) # his.npz为原始数据 (N,938,ci) ci=1
#     sample_list, sample_dict, sample_map = ptr1['sample_list'], ptr1['sample_dict'].item(), ptr1['sample_map'].item() # 字典，加上item()
#     # with open('data/hk/district13_roadindex.json', 'rb') as file: # 打开 JSON 文件
#     #     sd_district = pickle.load(file) # 读取文件内容, 字典
#     with open('data/hk/district13_roadindex.json', 'r') as file: # 打开 JSON 文件
#         district13_road_index = json.load(file) # 读取文件内容, 字典


#     # district_nodes = get_district_nodes(sample_dict, sample_map) # 初始化的区域字典，键0-12，值list(0-449)
#     total_nodes = sum(list(district13_road_index.values()), []) # 938个节点的list, 排序不一样[57,66,90,....]
#     node_count_global = {}
#     '''后面在选择时会更新node_count_global'''
#     node_count_global = calcute_global_dic(node_count_global, sample_list)

#     for i in range(7):
#         print(i)
#         print(sample_list)
#         # print(node_count_global)
#         print("-----------------------------------")
#         sample_list, node_count_global = optimize_selection(district13_road_index, 450, node_count_global)

    
#     print("finish!")



# if __name__ == "__main__":
#     main()
    