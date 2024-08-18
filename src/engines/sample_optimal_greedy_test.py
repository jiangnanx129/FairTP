
# coding: utf-8

'''
嵌套了香港数据集试验
这段代码有以下几个特点：

目标1的权重根据历史采样结果动态计算。具体来说，第一轮采样时，设目标1的权重为1；之后每轮采样结束后，根据上一轮采样结果的目标1和目标2的得分计算权重，采用指数衰减算法，使得历史结果对当前结果的影响越来越小。
每轮先选取每个区域中的一个节点，保证每个区域至少有一个节点被选择。
剩余的节点采用贪心算法选择，对于每个节点，计算它在目标1和目标2下的得分，然后按总得分从大到小排序，选取得分最高的节点加入采样结果中。
目标2的计算借助了历史采样结果。具体来说，对于当前要采样的节点，遍历历史采样结果，计算它在历史采样结果中的出现次数，并将其除以历史采样结果的长度得到平均出现次数，作为该节点的目标2得分。
希望这次回答能够满足您的要求，如果还有问题，请随时提问！
'''

# from src.engines.sample import *
import numpy as np
import os
import json
import random
import copy
import torch

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
        max_score_node = None
        max_score_region = None
        score_list = []
        
        for region, nodes in nodes_by_region_copy.items():
            if len(nodes) > 0:
                for node in nodes:
                    # score1 = abs(selected_counts[region] + 1 - mean_num) # mean_num为每个区域的平均数目, 差值越大越要选
                    # # 字典中获取键为 node 的值，如果字典中不存在该键，则返回默认值 0
                    # score2 = node_count_global.get(node, 0)  # 目标2的得分，前面这个节点被选择过的次数，数越大越少选（原来选过了）
                    # score_list.append((score1, score2, node, region))
                    '''只要输入数据》0，则到sigmoid》0.5'''
                    score1 = selected_counts[region] + 1 - mean_num # 越小越要选, 负数sigmoid后靠近0
                    score2 = node_count_global.get(node, 0)  # 越小越要选
                    score_list.append((score1, score2, node, region))


                    '''贪心算法'''
                    # w1 = 1                                                                      
                    # score = w1 * score1 + score2
                    # if score > max_score:
                    #     max_score = score
                    #     max_score_node = node
                    #     max_score_region = region

        # 如果不是所有元素相同. 就做正则化
        score1_list = list(zip(*score_list))[0]
        # print(score1_list)
        # pirnt()
        score2_list = list(zip(*score_list))[1]
        node_list = list(zip(*score_list))[2]
        region_list = list(zip(*score_list))[3]
    
        score1_tensor, score2_tensor = [torch.tensor(item) for item in score1_list], [torch.tensor(item) for item in score2_list]
        if is_all_elements_same(score1_tensor): # 所有元素一致，不归一化(nan),直接sigmoid
            score11 = torch.sigmoid(torch.tensor(score1_list))
        else: # 元素不一致，先归一化，再sigmoid
            s1_new_list, s1_new_tensor = normalize_list(score1_tensor) # input是list，里面每个元素是tensor
            score11 = torch.sigmoid(s1_new_tensor)
        if is_all_elements_same(score2_tensor):
            score22 = torch.sigmoid(torch.tensor(score2_list))
        else:
            s2_new_list, s2_new_tensor = normalize_list(score2_tensor)
            score22 = torch.sigmoid(s2_new_tensor)

        # print(score11, score11.shape)
        # print(score22, score22.shape)
        w1 = 1
        score = score11 * score22
        # print(score, score.shape)
        max_index = torch.argmin(score)
        # max_value = torch.max(tensor)
        # max_value, max_index = torch.max(score) # score是个tensor
        max_score_node = node_list[max_index.item()]
        max_score_region = region_list[max_index.item()]
        
        
        # if i==400:
        #     print(i, score_list)
        #     print("1: ",score1_list)
        #     print("-----------------------------------------")
        #     print("2: ",score2_list)
        #     print("+++++++++++++++++++++++++++++++++++++++++") # torch.tensor(a_values))) # new_tensor
            
        #     print("1:", score11)
        #     print("2:", score22)
        #     print(score)


        # if i==400:
            # print(i, score_list)
            # a_values = list(zip(*score_list))[0] # 取score_list中的score1
            # new_list, new_tensor = normalize_list([torch.tensor(item) for item in a_values])
            # print(a_values)
            # print("--------------",torch.sigmoid(torch.tensor(a_values))) # torch.tensor(a_values))) # new_tensor
            
            # a_values = list(zip(*score_list))[1] # 取score_list中的score1
            # new_list, new_tensor = normalize_list([torch.tensor(item) for item in a_values])
            # print(i, a_values)
            # print("--------------",torch.sigmoid(torch.tensor(a_values)))

        if max_score_region is None: # 某一区域的节点选完了！
            continue
        
        selected_nodes.append(max_score_node) # 添加一个节点
        '''更新了一个节点，更新selected_counts,键为区域，值为对应区域的采样数目'''
        if max_score_region not in selected_counts:
            selected_counts[max_score_region] = 0
        selected_counts[max_score_region] += 1

        nodes_by_region_copy[max_score_region].remove(max_score_node)
        '''更新了一个节点，更新node_count_global'''
        if max_score_node not in node_count_global:
            node_count_global[max_score_node] = 0
        node_count_global[max_score_node] += 1
        
        n_nodes_left -= 1
        i +=1
        # print("---------------------------------------",i) 

    return selected_nodes, node_count_global

# 判断list中所有元素是否一致
def is_all_elements_same(lst):
    arr = np.array(lst)
    return np.all(arr == arr[0])

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


'''把新的采样结果，sample_list添加到node_count_global字典中'''
def calcute_global_dic(node_count_global, sample_list):

    for node in sample_list:
        if node not in node_count_global:
            node_count_global[node] = 0
        node_count_global[node] += 1
    return node_count_global


def get_dataset_info(dataset):
    # base_dir = os.getcwd() + '/data/' # 不适合服务器
    base_dir ='/home/data/xjn/23largest_baseline/LargeST/data/'
    d = {
         'CA': [base_dir+'ca', base_dir+'ca/ca_rn_adj.npy', 8600],
         'GLA': [base_dir+'gla', base_dir+'gla/gla_rn_adj.npy', 3834],
         'GBA': [base_dir+'gba', base_dir+'gba/gba_rn_adj.npy', 2352],
        #  'SD': [base_dir+'sd', base_dir+'sd/sd_rn_adj.npy', 716],
         'SD': [base_dir+'sd', base_dir+'sd/2019/adj_initial350_all1.npy', 716], # 全1/0的矩阵
         'HK': [base_dir+'hk', base_dir+'hk/202010/adj_initial450_all1.npy', 938], 
        }
    assert dataset in d.keys()
    return d[dataset]





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

#     for i in range(3):
#         print(i)
#         print(sample_list)
#         print(node_count_global)
#         print("-----------------------------------")
#         sample_list = optimize_selection(district13_road_index, 450, node_count_global)

    
#     print("finish!")



# if __name__ == "__main__":
#     main()
    