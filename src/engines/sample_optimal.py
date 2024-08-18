# import torch
# import numpy as np
# import sys
# import os
# import time
# file_path = os.path.abspath(__file__) # /home/data/xjn/8fairness/src/engines/dcrnn_engine.py

# from src.base.engine import BaseEngine # special
# from src.utils.metrics import masked_mape, masked_rmse
# from src.utils.metrics import compute_all_metrics
# import random
# import torch.nn.functional as F
# '''
# 相较于sample.py文件，未考虑静态和动态损失影响的采样
# 使用了 minimize 函数求解优化问题
# 以上代码中使用了 minimize 函数求解优化问题，并提供了两个目标函数以及一个约束函数。
# 初始解生成函数使用了随机化的方法，可以通过多次运行获得不同的初始解，从而增加求解全局最优解的可能性。
# '''

# import numpy as np
# from scipy.optimize import minimize


# # 目标函数1：每个区域选择的节点数目的差额尽可能小
# def objective1(x, n_regions, region_size, nodes_per_region):
#     x = x.reshape(n_regions, -1)
#     selected_nodes_per_region = np.sum(x, axis=1)
#     max_diff = np.max(selected_nodes_per_region) - np.min(selected_nodes_per_region)
#     return max_diff

# # 目标函数1：每个区域选择的节点数目的差额尽可能小，且每个区域至少选一个节点
# '''计算每个区域的采样数目之间的方差，只要输入nodes_per_region就可以，得到一个数值为方差值'''
# def objective1(x, n_regions, region_size, nodes_per_region):
#     # x是节点列表0-937，n_regions是分区的数目，
#     # region_size表示每个区域可供选择的节点数，nodes_per_region表示每个区域要选的节点数目
#     selected_nodes_per_region = []
#     start_index = 0
#     for i in range(n_regions):
#         nodes = x[start_index:start_index + nodes_per_region[i]] # 都是顺序选择
#         selected_nodes_per_region.append(nodes)
#         start_index += region_size # 跨过当前区域，到下一个区域去选择
    
#     variances = np.var(selected_nodes_per_region, axis=1)
#     return np.sum(variances)



# # 目标函数2：同一区域内每个节点被选择的累计次数尽可能相等
# def objective2(x, n_regions, region_size, nodes_per_region):
#     x = x.reshape(n_regions, -1)
#     node_selection_counts = np.sum(x, axis=0)
#     n_nodes = node_selection_counts.shape[0]
#     n_selected_nodes_per_region = np.sum(x, axis=1)
#     target_count = np.mean(node_selection_counts)
#     node_variance = np.sum(np.square(node_selection_counts - target_count)) / n_nodes
#     region_variance = np.sum(np.square(n_selected_nodes_per_region - target_count)) / n_regions
#     return 0.5 * (node_variance + region_variance)

# # 目标函数2：同一区域内每个节点被选择的累计次数尽可能相等
# '''输入global的字典，计算'''
# def objective2(x, n_regions, region_size, nodes_per_region):
#     x = x.reshape(n_regions, -1)
#     node_selection_counts = np.sum(x, axis=0)
#     n_nodes = node_selection_counts.shape[0]
#     n_selected_nodes_per_region = np.sum(x, axis=1)
#     target_count = np.mean(node_selection_counts)
#     node_variance = np.sum(np.square(node_selection_counts - target_count)) / n_nodes
#     region_variance = np.sum(np.square(n_selected_nodes_per_region - target_count)) / n_regions
#     return 0.5 * (node_variance + region_variance)



# # 约束函数：每次选择30个节点
# def constraint(x, n_regions, region_size, nodes_per_region):
#     return np.sum(x) - 30 * n_regions


# # 初始解生成函数
# def generate_initial_solution(n_regions, region_size, nodes_per_region):
#     x0 = np.zeros((n_regions, region_size))
#     for i in range(n_regions):
#         idx = np.random.choice(region_size, size=nodes_per_region[i], replace=False)
#         x0[i][idx] = 1
#     x0 = x0.reshape(-1)
#     return x0


# # 定义优化问题参数
# n_regions = 10
# region_size = 5
# nodes_per_region = [3, 4, 2, 1, 5, 2, 3, 2, 2, 6]

# # 约束条件
# cons = {
#     'type': 'eq',
#     'fun': constraint,
#     'args': (n_regions, region_size, nodes_per_region),
# }

# # 初始化解
# x0 = generate_initial_solution(n_regions, region_size, nodes_per_region)

# # 求解最小化问题
# results = []
# for i in range(3):
#     res = minimize(lambda x: objective1(x, n_regions, region_size, nodes_per_region) +
#                                objective2(x, n_regions, region_size, nodes_per_region),
#                    x0=x0, constraints=cons, method='SLSQP', options={'maxiter': 10000})
#     x = np.round(res.x.reshape(n_regions, -1))
#     selected_nodes_per_region = np.sum(x, axis=1)
#     node_selection_counts = np.sum(x, axis=0)
#     results.append((selected_nodes_per_region, node_selection_counts))
#     x0 = x.reshape(-1)

# # 输出结果
# for i, result in enumerate(results):
#     print(f"第 {i+1} 次选择:")
#     print(f"每个区域选择的节点数目: {result[0]}")
#     print(f"每个节点被选择的累计次数: {result[1]}")










# import numpy as np
# from scipy.optimize import minimize


# # 目标函数1：每个区域选择的节点数目的差额尽可能小，且每个区域至少选一个节点
# def objective1(x, n_regions, nodes_per_region):
#     selected_nodes_per_region = []
#     idx = 0
#     for n_nodes in nodes_per_region:
#         selected_nodes_per_region.append(int(np.sum(x[idx:idx+n_nodes])))
#         idx += n_nodes
#     max_diff = np.max(selected_nodes_per_region) - np.min(selected_nodes_per_region)
#     min_select = np.min(selected_nodes_per_region)
#     # 判断每个区域是否至少选了一个节点
#     if np.any(selected_nodes_per_region == 0):
#         return np.inf
#     return max_diff


# # 目标函数2：同一区域内每个节点被选择的累计次数尽可能相等
# def objective2(x, n_regions, nodes_per_region):
#     selected_nodes_per_region = []
#     idx = 0
#     for n_nodes in nodes_per_region:
#         selected_nodes_per_region.append(int(np.sum(x[idx:idx+n_nodes])))
#         idx += n_nodes
#     node_selection_counts = np.array(selected_nodes_per_region)
#     n_nodes = node_selection_counts.shape[0]
#     target_count = np.mean(node_selection_counts)
#     node_variance = np.sum(np.square(node_selection_counts - target_count)) / n_nodes
#     return node_variance


# # 约束函数：每次选择30个节点
# def constraint(x, n_regions, total_nodes):
#     return np.sum(x) - total_nodes


# # 初始解生成函数
# def generate_initial_solution(n_regions, nodes_per_region):
#     total_nodes = sum(nodes_per_region) # 30
#     x0 = np.zeros(total_nodes)
#     idx = 0
#     for i in range(n_regions):
#         n_nodes = nodes_per_region[i]
#         x0[idx:idx+n_nodes] = 1
#         idx += n_nodes
#     return x0


# # 定义优化问题参数
# n_regions = 10 # 10个区域，一共有50个节点
# nodes_per_region = [3, 4, 2, 1, 5, 2, 3, 2, 2, 6] # 每个区域要采样额节点数目，一共30
# total_nodes = sum(nodes_per_region) # 30

# # 约束条件
# cons = {
#     'type': 'eq', # 表示这是一个等式约束
#     'fun': constraint, # 一个函数对象，指定了约束函数 constraint
#     'args': (n_regions, total_nodes), # 约束函数的额外参数，即 (n_regions, total_nodes)
# }

# # 初始化解
# x0 = generate_initial_solution(n_regions, nodes_per_region)

# # 求解最小化问题
# results = []
# for i in range(3):
#     # 在调用优化函数时，可以将约束条件作为参数传递给优化函数，以确保求解过程中满足约束条件：constraints=cons 指定了约束条件为上述定义的 cons。这样，在求解过程中，优化函数会自动考虑约束条件，以找到满足约束条件的最优解。
#     res = minimize(lambda x: objective1(x, n_regions, nodes_per_region) +
#                                objective2(x, n_regions, nodes_per_region),
#                    x0=x0, constraints=cons, method='SLSQP', options={'maxiter': 10000})
#     x = np.round(res.x)
#     selected_nodes_per_region = []
#     idx = 0
#     for n_nodes in nodes_per_region:
#         selected_nodes_per_region.append(int(np.sum(x[idx:idx+n_nodes])))
#         idx += n_nodes
#     results.append(selected_nodes_per_region)
#     x0 = x

# 输出结果
# for i, result in enumerate(results):
#     print(f"第 {i+1} 次选择:")
#     print(f"每个区域选择的节点数目: {result}")

import numpy as np
from scipy.optimize import minimize
import json


# 目标函数1：每个区域选择的节点数目的差额尽可能小，且每个区域至少选一个节点
def objective1(x, n_regions, nodes_per_region):
    selected_nodes_per_region = []
    idx = 0
    for n_nodes in nodes_per_region:
        selected_nodes_per_region.append(int(np.sum(x[idx:idx+n_nodes])))
        idx += n_nodes
    max_diff = np.max(selected_nodes_per_region) - np.min(selected_nodes_per_region)
    min_select = np.min(selected_nodes_per_region)
    # 判断每个区域是否至少选了一个节点
    if np.any(selected_nodes_per_region == 0):
        return np.inf
    return max_diff


# 目标函数2：同一区域内每个节点被选择的累计次数尽可能相等
def objective2(x, n_regions, nodes_per_region):
    selected_nodes_per_region = []
    idx = 0
    for n_nodes in nodes_per_region:
        selected_nodes_per_region.append(int(np.sum(x[idx:idx+n_nodes])))
        idx += n_nodes
    node_selection_counts = np.array(selected_nodes_per_region)
    n_nodes = node_selection_counts.shape[0]
    target_count = np.mean(node_selection_counts)
    node_variance = np.sum(np.square(node_selection_counts - target_count)) / n_nodes
    return node_variance


# 约束函数：每次选择select_nodes个节点
def constraint(x, select_nodes):
    return np.sum(x) - select_nodes


# 初始解生成函数
def generate_initial_solution(total_nodes, select_nodes):
    x0 = np.zeros(total_nodes)
    x0[:select_nodes] = 1  # 初始化前select_nodes个节点被选择
    return x0


# 定义优化问题参数
total_nodes = 50
select_nodes = 30
region_dict = {
  1: [0, 1, 2, 3, 4],
  2: [5, 6, 7, 8, 9],
  3: [10, 11, 12, 13],
  4: [14, 15, 16, 17, 18],
  5: [19, 20, 21, 22],
  6: [23, 24, 25, 26, 27],
  7: [28, 29, 30, 31, 32],
  8: [33, 34, 35, 36, 37],
  9: [38, 39, 40, 41],
  10: [42, 43, 44, 45, 46, 47, 48, 49]
}
nodes_per_region = [len(region_dict[i]) for i in range(1, len(region_dict)+1)]

# 约束条件
cons = {
    'type': 'eq',
    'fun': constraint,
    'args': (select_nodes,),
}

# 初始化解
x0 = generate_initial_solution(total_nodes, select_nodes)

for i in range(3):
    # 在调用优化函数时，可以将约束条件作为参数传递给优化函数，以确保求解过程中满足约束条件：constraints=cons 指定了约束条件为上述定义的 cons。这样，在求解过程中，优化函数会自动考虑约束条件，以找到满足约束条件的最优解。
    res = minimize(lambda x: objective1(x, len(region_dict), nodes_per_region) + objective2(x, len(region_dict), nodes_per_region),
               x0=x0, constraints=cons, method='SLSQP', options={'maxiter': 10000})
    x = np.round(res.x)
    selected_nodes = np.where(x == 1)[0]
    x0 = x.reshape(-1)
    print(f"第 {i+1} 次选择:")
    print(f"每个区域选择的节点数目: {selected_nodes}")


with open('data/hk/district13_roadindex.json', 'r') as file: # 打开 JSON 文件
    district13_road_index = json.load(file) # 读取文件内容, 字典
print(district13_road_index)