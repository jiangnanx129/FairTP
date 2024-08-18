'''
优化问题的解决，python，有50个节点属于10个区域，每个区域的节点数目不一样，有分区字典指示,键为区域，值为该区域包含的节点列表。
根据两个目标的最小化，每次从这10个区域中一共选出30个节点，该选择进行3次，每次都要两个目标最小化。
目标1是每个区域选择的节点数目的差额尽可能小，目标2是同一区域内每个节点被选择的累计次数尽可能相等（确保选择的多样性）
'''

'''
多目标优化，循环遍历求解
在这个示例中，nodes_by_region表示包含每个区域及其对应的节点列表的字典。total_nodes则是所有节点的列表。
num_selections表示要进行多少次选择，每次选择30个节点。最后，optimize_selection函数返回了每次选择的结果列表。
该解决方案使用两个循环来遍历选择组合，并计算目标1和目标2的值。在每次选择中，会更新节点被选择的次数，并将得到的最优选择添加到结果列表中。
'''
import itertools
from collections import defaultdict
from src.engines.sample import *

# 目标函数1：每个区域选择的节点数目的差额尽可能小
def objective1(sample_dict, selected_nodes, node_count_global): # 采样字典，采样元组(相当于list)
    node_count_global2 = node_count_global.copy()  # 使用 copy() 方法复制字典
    # node_count_global2 = node_count_global # 避免改变node_count_global
    mean_num = int(len(selected_nodes)/len(list(sample_dict.keys()))) # 450/13=平均采样数目，每个区域一样的数目
    selected_nodes_per_region = []
    diff = 0 # 原始方差
    for district_id, nodes in sample_dict.items(): # 循环采样好的字典，值0-937
        selected_nodes_per_region.append(len(nodes)) # 每个区域采样的节点数目
        diff += (len(nodes) - mean_num)**2 # 节点数目的方差

        # 帮忙统计节点的采样次数，方便objective2的计算！
        for node in nodes: # nodes取值0-937
            if node not in node_count_global2:
                node_count_global2[node] = 0
            node_count_global2[node] +=1 # 采样计数

    return diff, node_count_global2

# 目标函数2：每个区域选择的节点数目的差额尽可能小
def objective2(node_count_global, nodes_by_region): 
    '''
    node_count_global: 全局采样次数统计字典，键为1-50，值为对应节点采样的次数
    nodes_by_region: 所有节点的分区字典，键为区域A-J，值为list，每个区域对应的节点
    '''
    diff = 0 # 原始
    ever_sampe_nodes = list(node_count_global.keys())
    # 针对每个区域看采样偏差
    for district_id, nodes in nodes_by_region.items():
        region_sampe_nodes = [] # 某一区域采样过的节点次数
        for node in nodes: # node
            if node in ever_sampe_nodes: # 该区域采样过的所有节点
                region_sampe_nodes.append(node_count_global[node]) # add的是对应节点的采样次数
            else:
                region_sampe_nodes.append(0) # 没采样的节点次数为0

        region_diff = calculate_sum_of_differences(region_sampe_nodes)
        diff += region_diff

    return diff

# 计算差距，输入list
def calculate_sum_of_differences(lst):
    sum_of_differences = 0
    
    for i in range(len(lst)):
        for j in range(i+1, len(lst)):
            difference = abs(lst[i] - lst[j])
            sum_of_differences += difference
    
    return sum_of_differences


def optimize_selection(nodes_by_region, total_nodes, sample_num, node_count_global, n_number):
    '''
    nodes_by_region：所有节点分区情况，937个节点
    total_nodes：所有节点的列表[0,1,2,...,937]
    sample_num：采样总数450
    node_count_global：字典统计每个节点被选择的次数，键0-937，值为对应节点的采样次数
    n_number: 循环次数，为了提高效率
    '''
    
    # 每次选择时的目标1和目标2的值
    best_obj1_diff = float('inf') # 每个区域采样数目尽可能一致
    best_obj2_diff = float('inf') # 同一区域每个节点的采样次数尽可能一致
    best_selection = None
    counts = 0
    # 遍历所有的选择组合,937中sample出450的所有组合
    '''效率太低！就从200个情况中选择吧， for循环替换为while'''
    # for selected_nodes in itertools.combinations(total_nodes, sample_num): # 它可以生成从总节点集合 total_nodes 中选择 sample_num 个节点的所有组合。
    while counts < n_number:
        selected_nodes = random.sample(total_nodes, sample_num)
        
        #sample_map = sum_map(selected_nodes, sample_num) # 字典，键0-449，值0-937
        sample_dict = sample_map_district(selected_nodes, nodes_by_region) # 采样的节点元组分配到区域，返回字典，键为区域，值为对应的list保存采样的元素
        #district_nodes = get_district_nodes(sample_dict, sample_map)

        if len(list(sample_dict.keys())) != len(list(nodes_by_region.keys())): # 每个区域都有被采样到
            continue # 不是每个区域都有采样，跳过本次循环
        
        # 根据新的采样结果更新node_count_global字典，键为节点0-937，值为节点采样过的次数

        # 计算当前采样方式的两个目标值
        obj1_diff, node_count_global2 = objective1(sample_dict, selected_nodes, node_count_global) # input两个字典,采样的元组
        obj2_diff = objective2(node_count_global2, nodes_by_region)
        
        # 更新最优选择
        if obj1_diff < best_obj1_diff or (obj1_diff == best_obj1_diff and obj2_diff < best_obj2_diff):
            best_obj1_diff = obj1_diff
            best_obj2_diff = obj2_diff
            best_selection = selected_nodes
        counts +=1
        # print(node_count_global)
        # print("aaa")

    '''在engine文件中计算个体动态损失时更新了node_count_global'''
    # # 更新节点被选择的次数并添加到结果列表中
    # for node in best_selection: # 循环采样的list
    #     if node not in node_count_global:
    #         node_count_global[node] = 0
    #     node_count_global[node] +=1 # 更新采样计数
    # print("--------------------------")

    return best_selection # , node_count_global # 返回采样结果！最优的


# 区域与节点的字典表示
nodes_by_region = {
    "A": [1, 2, 3, 4, 5],
    "B": [6, 7, 8, 9],
    "C": [10, 11, 12, 13, 14, 15],
    "D": [16, 17, 18],
    "E": [19, 20, 21, 22, 23],
    "F": [24, 25, 26, 27],
    "G": [28, 29, 30, 31, 32],
    "H": [33, 34, 35, 36],
    "I": [37, 38, 39, 40, 41, 42],
    "J": [43, 44, 45, 46, 47, 48, 49, 50]
}

# # result = sum(list(nodes_by_region.values()), [])
# # print(result)
# # pirnt()

# # 所有节点的列表
# total_nodes = list(range(1, 51))

# # 进行3次选择，每次选择30个节点
# num_selections = 3
# sample_num = 30
# node_count_global = {}
# # # 调用优化函数
# # best_selection, node_count_global = optimize_selection(nodes_by_region, total_nodes, sample_num, node_count_global)

# # 打印结果
# for i in range(3):
#     print(node_count_global)
#     # 调用优化函数
#     best_selection, node_count_global = optimize_selection(nodes_by_region, total_nodes, sample_num, node_count_global,5)
#     print(f"第{i+1}次选择：{best_selection}")
