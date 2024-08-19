
# coding: utf-8

'''
嵌套了香港数据集试验
这段代码有以下几个特点：

目标1的权重根据历史采样结果动态计算。具体来说，第一轮采样时，设目标1的权重为1；之后每轮采样结束后，根据上一轮采样结果的目标1和目标2的得分计算权重，采用指数衰减算法，使得历史结果对当前结果的影响越来越小。
每轮先选取每个区域中的一个节点，保证每个区域至少有一个节点被选择。
剩余的节点采用贪心算法选择，对于每个节点，计算它在目标1和目标2下的得分，然后按总得分从大到小排序，选取得分最高的节点加入采样结果中。
目标2的计算借助了历史采样结果。具体来说，对于当前要采样的节点，遍历历史采样结果，计算它在历史采样结果中的出现次数，并将其除以历史采样结果的长度得到平均出现次数，作为该节点的目标2得分。

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
# nodes_by_region每个batch一个，键为0-938，值为采样次数
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

        '''更新了一个节点，更新node_count_global'''
        if node not in node_count_global:
            node_count_global[node] = 0
        node_count_global[node] += 1
           
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





###########################################################################################################################
'''概率采样，区域平衡采样不能实现'''
# sorted_dict = pro_sample2(values_region, values_node, self.district13_road_index)
# sample_list = resampel(sorted_dict,self.sample_num,self.district13_road_index)

# 除了这两个 还要区域采样计数字典（没办法算 还没开始采样），和节点采样计数字典node_count_global  3412
# 用dynamic_cal5_global_T4(yl_global, node_count_global, district13_road_index, device)
def pro_sample2(values_region, values_node, values_node_count_global, district13_road_index):
    '''
    node_count_global字典：键为0-938，值为该节点在当前epoch内已经有的采样次数
    values_node_count_global，形式和values_node一样，一起得到的
    '''
    
    node_cs = {}
    for v,node_list in district13_road_index.items(): # 键0-12，值0-938
        for node in node_list: # 循环该区域的节点列表：
            # node_cs[node] = 0.001*mape_for_each_region[int(v)] * condition_for_each_node[node]
            node_cs[node] = values_region[int(v)] * values_node[node] * values_node_count_global[node]

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

###########################################################################################################################







'''3.每个batch根据123，每T个batch根据1234'''
def optimize_selection_T_loss34(nodes_by_region, sample_num, node_count_global, values_region, values_node, device ):
    '''
    nodes_by_region：所有节点分区情况，938个节点
    total_nodes：所有节点的列表[0,1,2,...,937]
    sample_num：采样总数450
    node_count_global：一个epoch一个。字典统计每个节点被选择的次数，键0-937，值为对应节点的采样次数
    region_value:
    indi_value
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

        '''更新了一个节点，更新node_count_global'''
        if node not in node_count_global:
            node_count_global[node] = 0
        node_count_global[node] += 1
           
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
                score1_list2.append(values_region[int(region)])
            else:
                continue

        '''先选择一个区域'''
        result1 = process_tensor(torch.tensor(score1_list), device) # 返回归一化的区域参数，每个区域距离均值的距离，越小越要采样
        result2 = torch.stack(score1_list2) # 保持list中每个元素的梯度，有梯度的
        result12 = result1*result2
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


'''3.每个batch根据12，每T个batch根据124'''
def optimize_selection_T_loss4(nodes_by_region, sample_num, node_count_global, values_node, device ):
    '''
    nodes_by_region：所有节点分区情况，938个节点
    total_nodes：所有节点的列表[0,1,2,...,937]
    sample_num：采样总数450
    node_count_global：一个epoch一个。字典统计每个节点被选择的次数，键0-937，值为对应节点的采样次数
    region_value:
    indi_value
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

        '''更新了一个节点，更新node_count_global'''
        if node not in node_count_global:
            node_count_global[node] = 0
        node_count_global[node] += 1
           
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
            score2_list2.append(values_node[node]) # node取值0-938

        result3 = process_tensor(torch.tensor(score2_list), device) # 返回归一化的区域参数，每个区域距离均值的距离，越小越要采样
        result4 = torch.stack(score2_list2)
        result34 = result3*result4
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

# 14采样
def optimize_selection_T_134(nodes_by_region, sample_num, values_region, values_node, device ):
    '''
    nodes_by_region：所有节点分区情况，938个节点
    total_nodes：所有节点的列表[0,1,2,...,937]
    sample_num：采样总数450
    node_count_global：一个epoch一个。字典统计每个节点被选择的次数，键0-937，值为对应节点的采样次数
    region_value:
    indi_value
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

        # '''更新了一个节点，更新node_count_global'''
        # if node not in node_count_global:
        #     node_count_global[node] = 0
        # node_count_global[node] += 1
           
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
                score1_list2.append(values_region[int(region)])
            else:
                continue

        '''先选择一个区域'''
        result1 = process_tensor(torch.tensor(score1_list), device) # 返回归一化的区域参数，每个区域距离均值的距离，越小越要采样
        result2 = torch.stack(score1_list2) # 保持list中每个元素的梯度，有梯度的
        result12 = result1*result2
        region_min_index = torch.argmin(result12)
        selected_region = region_list[region_min_index.item()]

        if selected_region is None: # 某一区域的节点选完了！双重保险
            continue

        '''选节点,确定区域后'''
        nodes_list = nodes_by_region_copy[selected_region]
        for node in nodes_list:
            # score2 = node_count_global.get(node, 0)  # 同一区域的节点，越小越要选
            # score2_list.append(score2)
            score2_list2.append(values_node[node]) # node取值0-938

        #result3 = process_tensor(torch.tensor(score2_list), device) # 返回归一化的区域参数，每个区域距离均值的距离，越小越要采样
        result4 = torch.stack(score2_list2)
        #result34 = result3*result4
        node_min_index = torch.argmin(result4)
        selected_node = nodes_list[node_min_index.item()]

        
        selected_nodes.append(selected_node) # 添加一个节点
        '''更新了一个节点，更新selected_counts,键为区域，值为对应区域的采样数目'''
        if selected_region not in selected_counts:
            selected_counts[selected_region] = 0
        selected_counts[selected_region] += 1

        nodes_by_region_copy[selected_region].remove(selected_node) # 删除某区域中选择过的节点

        # '''更新了一个节点，更新node_count_global'''
        # if selected_node not in node_count_global:
        #     node_count_global[selected_node] = 0
        # node_count_global[selected_node] += 1
        
        n_nodes_left -= 1
        i +=1

    return selected_nodes

def optimize_selection_T_134_equal(nodes_by_region, sample_num, values_region, values_node, device ):
    '''
    nodes_by_region：所有节点分区情况，938个节点
    total_nodes：所有节点的列表[0,1,2,...,937]
    sample_num：采样总数450
    node_count_global：一个epoch一个。字典统计每个节点被选择的次数，键0-937，值为对应节点的采样次数
    region_value:
    indi_value
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

        # '''更新了一个节点，更新node_count_global'''
        # if node not in node_count_global:
        #     node_count_global[node] = 0
        # node_count_global[node] += 1
           
    n_nodes_left = sample_num - len(selected_nodes) # 每个区域选择一个后，剩下的可选择次数
    
    i =1
    # 使用贪心算法选择剩余的节点
    while n_nodes_left > 0:
        max_score = float('-inf')
        selected_region = None
        selected_node = None
        
        score_list = []
        

        # 先区域，再节点
        for region, nodes in nodes_by_region_copy.items():
            '''选区域'''
            if len(nodes) > 0: # 确保区域还有节点，没有节点的话，socre1持续为负数但无法取值，会陷入死循环
                for node in nodes:

                    score1 = selected_counts[region] + 1 - mean_num # 越小越要选, 负数sigmoid后靠近0
                    score2 = values_node[node] # 已经是tensor
                    score3 = values_region[int(region)]
                    score_list.append((score1, score2, node, region, score3))
            
            else:
                continue
        
        score1_list = list(zip(*score_list))[0]
        score2_list = list(zip(*score_list))[1] # tensor
        score3_list = list(zip(*score_list))[4] # tensor
        nodes_list = list(zip(*score_list))[2]
        region_list = list(zip(*score_list))[3]
        result1 = process_tensor(torch.tensor(score1_list), device) # 返回归一化的区域参数，每个区域距离均值的距离，越小越要采样
        result2 = torch.stack(score2_list)
        result3 = torch.stack(score3_list)
        
        # process_tensor(torch.tensor(score2_list), device) # 返回归一化的区域参数，每个区域距离均值的距离，越小越要采样
        
        score = result1 * result2 * result3
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

        # '''更新了一个节点，更新node_count_global'''
        # if selected_node not in node_count_global:
        #     node_count_global[selected_node] = 0
        # node_count_global[selected_node] += 1
        
        n_nodes_left -= 1
        i +=1

    return selected_nodes



# 14采样
def optimize_selection_T_14(nodes_by_region, sample_num, values_node, device ):
    '''
    nodes_by_region：所有节点分区情况，938个节点
    total_nodes：所有节点的列表[0,1,2,...,937]
    sample_num：采样总数450
    node_count_global：一个epoch一个。字典统计每个节点被选择的次数，键0-937，值为对应节点的采样次数
    region_value:
    indi_value
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

        # '''更新了一个节点，更新node_count_global'''
        # if node not in node_count_global:
        #     node_count_global[node] = 0
        # node_count_global[node] += 1
           
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
                # score1_list2.append(values_region[int(region)])
            else:
                continue

        '''先选择一个区域'''
        result1 = process_tensor(torch.tensor(score1_list), device) # 返回归一化的区域参数，每个区域距离均值的距离，越小越要采样
        #result2 = torch.stack(score1_list2) # 保持list中每个元素的梯度，有梯度的
        #result12 = result1*result2
        region_min_index = torch.argmin(result1)
        selected_region = region_list[region_min_index.item()]

        if selected_region is None: # 某一区域的节点选完了！双重保险
            continue

        '''选节点,确定区域后'''
        nodes_list = nodes_by_region_copy[selected_region]
        for node in nodes_list:
            # score2 = node_count_global.get(node, 0)  # 同一区域的节点，越小越要选
            # score2_list.append(score2)
            score2_list2.append(values_node[node]) # node取值0-938

        #result3 = process_tensor(torch.tensor(score2_list), device) # 返回归一化的区域参数，每个区域距离均值的距离，越小越要采样
        result4 = torch.stack(score2_list2)
        #result34 = result3*result4
        node_min_index = torch.argmin(result4)
        selected_node = nodes_list[node_min_index.item()]

        
        selected_nodes.append(selected_node) # 添加一个节点
        '''更新了一个节点，更新selected_counts,键为区域，值为对应区域的采样数目'''
        if selected_region not in selected_counts:
            selected_counts[selected_region] = 0
        selected_counts[selected_region] += 1

        nodes_by_region_copy[selected_region].remove(selected_node) # 删除某区域中选择过的节点

        # '''更新了一个节点，更新node_count_global'''
        # if selected_node not in node_count_global:
        #     node_count_global[selected_node] = 0
        # node_count_global[selected_node] += 1
        
        n_nodes_left -= 1
        i +=1

    return selected_nodes

# 14采样，但是14同级
def optimize_selection_T_14_equal(nodes_by_region, sample_num, values_node, device ):
    '''
    nodes_by_region：所有节点分区情况，938个节点
    total_nodes：所有节点的列表[0,1,2,...,937]
    sample_num：采样总数450
    node_count_global：一个epoch一个。字典统计每个节点被选择的次数，键0-937，值为对应节点的采样次数
    region_value:
    indi_value
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

        # '''更新了一个节点，更新node_count_global'''
        # if node not in node_count_global:
        #     node_count_global[node] = 0
        # node_count_global[node] += 1
           
    n_nodes_left = sample_num - len(selected_nodes) # 每个区域选择一个后，剩下的可选择次数
    
    i =1
    # 使用贪心算法选择剩余的节点
    while n_nodes_left > 0:
        max_score = float('-inf')
        selected_region = None
        selected_node = None
        
        score_list = []
        

        # 先区域，再节点
        for region, nodes in nodes_by_region_copy.items():
            '''选区域'''
            if len(nodes) > 0: # 确保区域还有节点，没有节点的话，socre1持续为负数但无法取值，会陷入死循环
                for node in nodes:

                    score1 = selected_counts[region] + 1 - mean_num # 越小越要选, 负数sigmoid后靠近0
                    score2 = values_node[node] # 已经是tensor
                    score_list.append((score1, score2, node, region))
            
            else:
                continue
        
        score1_list = list(zip(*score_list))[0]
        score2_list = list(zip(*score_list))[1] # tensor
        nodes_list = list(zip(*score_list))[2]
        region_list = list(zip(*score_list))[3]
        result1 = process_tensor(torch.tensor(score1_list), device) # 返回归一化的区域参数，每个区域距离均值的距离，越小越要采样
        result2 = torch.stack(score2_list)
        
        # process_tensor(torch.tensor(score2_list), device) # 返回归一化的区域参数，每个区域距离均值的距离，越小越要采样
        
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

        # '''更新了一个节点，更新node_count_global'''
        # if selected_node not in node_count_global:
        #     node_count_global[selected_node] = 0
        # node_count_global[selected_node] += 1
        
        n_nodes_left -= 1
        i +=1

    return selected_nodes




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





'''YES:把新的采样结果，sample_list添加到node_count_global字典中, dynamic_cal5_global_T4'''
def calcute_global_dic(node_count_global, sample_list):

    for node in sample_list:
        if node not in node_count_global:
            node_count_global[node] = 0
        node_count_global[node] += 1
    return node_count_global


'''统计区域的采样结果，用不到！还没开始采样 算区域没有意义'''
# district_nodes：键0-12，值list 节点列表0-449
def calcute_batch_dic(district_nodes):
    # 根据上一次的采样情况
    counts_region_list = []
    for node_list in list(district_nodes.values()): # 循环每个区域
        counts_region_list.append(len(node_list))

    return counts_region_list # list









# 初始分层采样, input(b,t,n,c)-(t,n,c), output(t,n',c)
def initial_sample(sd_district, sam_num): # 键区域0-13，值list0-938
    
    sum_node_list = list(sd_district.values()) # 双层list
    sum_node = sum(len(sublist) for sublist in sum_node_list) # 总结点数
    # 计算每个区域的节点数目占总节点数目的比例, region_nodes是list, float，region是(3,4)
    region_proportions = {region: len(region_nodes) / sum_node for region, region_nodes in sd_district.items()}
    # 根据比例计算每个区域应该抽取的节点数目
    region_samples = {region: round(proportion * sam_num + 0.5) for region, proportion in region_proportions.items()}
    
    zero_keys = [key for key, value in region_samples.items() if value == 0]
    count = len(zero_keys)
    
    sample = []
    for region, sample_count in region_samples.items():
        # 随机抽样，从几个节点中随机选！不适用于后面采样，只适用于初始化采样
        region_node_list = sd_district[region] # 某个区域的节点列表
        # 随机采样，后面可以换成切片！分数高的在前/后，直接切片
        nodes_sample = random.sample(region_node_list, sample_count)
        sample.extend(nodes_sample)
        # sample.extend(np.random.choice(nodes_by_region[region], size=sample_count, replace=False))

    # 如果由于四舍五入的原因，抽取的节点数目少于sum，就从剩余的节点中随机抽取
    remaining_samples_needed = sam_num - len(sample) # 少采了！
    flat_node_list = [num for sublist in sum_node_list for num in sublist] # 展开2维list
    if remaining_samples_needed > 0:
        remaining_nodes = [node for node in flat_node_list if node not in sample]
        sample.extend(random.sample(remaining_nodes, remaining_samples_needed))
        # sample.extend(np.random.choice(remaining_nodes, size=remaining_samples_needed, replace=False))
    if remaining_samples_needed < 0: # 如果超过了
        a = len(sample) - sam_num
        sample = []
        longest_list_key = max(region_samples, key=lambda k: region_samples[k]) # 找到最大的采样区域
        # 重新采样，原本采样最多的少采一点
        for region, sample_count in region_samples.items():
            region_node_list = sd_district[region]
            if region == longest_list_key:
                nodes_sample = random.sample(region_node_list, sample_count-a) # 在采样最多的区域少采一点！
                sample.extend(nodes_sample)
                continue
            nodes_sample = random.sample(region_node_list, sample_count)
            sample.extend(nodes_sample)

    return sample # 采样的列表，每个元素为节点的index（0-937）
