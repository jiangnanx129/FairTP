
# coding: utf-8
# '''
# 动态自适应采样。有50个节点不平衡的分布在10个区域，每个区域的节点数目不一样，有分区字典指示,键为区域，值为该区域包含的节点列表，
# 每轮从这10个区域中一共选出30个节点，并保证每个区域至少有一个采样到，每次采样都要两个目标最小化。
# 目标1是每个区域选择的节点数目的差额尽可能小，目标2是同一区域内每个节点被选择的累计次数尽可能相等（确保选择的多样性），
# 注意，在目标2中，上一次的采样结果会影响下一次的采样。
# '''

# '''
# 针对这个动态自适应采样的需求，可以使用以下步骤来实现：

# 初始化：根据给定的分区字典，计算每个区域的节点数目，并记录每个节点的累计次数。

# 选择初始节点：从每个区域中选择一个节点，确保每个区域至少有一个节点被选中，并更新节点的累计次数。

# 选择余下节点：对于剩下需要选择的节点数量（总共30个减去已选择的节点数），进行迭代选择。

# 3.1 计算节点选择概率：根据目标1和目标2计算每个节点被选择的概率。目标1可以通过计算每个区域内节点数目的差值来得到，目标2可以使用节点的累计次数进行调整。

# 3.2 根据概率选择节点：根据计算得到的节点选择概率，从所有节点中选择一个节点。

# 3.3 更新节点的累计次数：更新选中节点的累计次数，以便影响下一轮的采样结果。

# 返回采样结果：最终得到的30个节点即为一轮的采样结果。

# 在每一轮的采样中，目标1和目标2的权重可以进行调整，以达到更好的平衡效果。

# 需要注意的是，上一次的采样结果会影响下一次的采样，可以通过控制目标2中的节点累计次数来实现。如果需要保持较大的多样性，可以适当降低目标2的权重。

# 此外，由于需要进行多轮的采样，还可以根据每轮的结果动态调整区域之间的节点分布，以进一步优化采样效果。

# 以上是一个简单的实现思路，具体的实现细节可以根据具体需求和算法框架进行调整和完善
# '''


# '''
# 可以使用基于多目标优化的进化算法来实现上述的动态自适应采样，比如NSGA-II（Non-dominated Sorting Genetic Algorithm II）等。NSGA-II算法是一种常用的进化算法，能够有效地解决多目标优化问题。

# 具体而言，可以将每个节点的选择概率作为一个目标，将采样结果的目标1和目标2作为另外两个目标，将这三个目标组合成一个多目标优化问题，使用NSGA-II算法求解最优化的节点选择方案。

# 在NSGA-II算法中，每个个体都表示成一个状态，对于动态自适应采样问题，可以将每个状态表示为一个节点选择方案。然后通过交叉、变异等操作进行种群演化，并使用非支配排序和拥挤度距离等技术保持种群多样性。

# 为了实现多目标优化，在NSGA-II算法中需要使用非支配排序和拥挤度距离等技术。非支配排序首先根据各个目标函数的大小关系进行排序，将支配其他个体的排在前面，然后根据支配等级和拥挤度距离确定个体的适应度值，以保持可行解的多样性。拥挤度距离是指一个个体周围的个体密度，即距离该个体最近的其他个体之间的距离平均值。

# 通过使用NSGA-II算法，可以实现动态自适应采样问题的多目标优化，同时保持各个目标之间的平衡性。
# '''


# '''
# 对于高效地实现上述的动态自适应采样问题，可以使用一些基于局部搜索和启发式优化的算法，如模拟退火算法、遗传算法、禁忌搜索等。

# 其中，模拟退火算法是一种基于随机漫步的全局优化算法，通过不断地接受劣解以避免陷入局部最优解。在动态自适应采样问题中，可以将节点选择方案作为状态空间，并根据目标1和目标2定义能量函数，用模拟退火算法在状态空间中搜索全局最优解。

# 遗传算法是一种模拟生物进化过程的启发式优化算法，通过交叉和变异等操作进行种群演化，逐步优化个体的适应度值。在动态自适应采样问题中，可以将每个节点的选择概率作为一个基因，使用遗传算法进行优化，以求得最优的节点选择方案。

# 禁忌搜索是一种局部搜索算法，通过记录某些状态的历史信息来避免搜索到相似的状态，以避免陷入局部最优解。在动态自适应采样问题中，可以将节点选择方案作为状态空间，并根据目标1和目标2定义目标函数，使用禁忌搜索算法在状态空间中搜索全局最优解。

# 这些算法都可以较高效地解决动态自适应采样问题，但每种算法都在某些案例中表现更优，需要根据具体场景进行选择和调整。
# '''

import random

# 定义分区字典，键为区域，值为该区域包含的节点列表
# nodes_by_region = {
#     "A": [1, 2, 3, 4, 5],
#     "B": [6, 7, 8, 9],
#     "C": [10, 11, 12, 13, 14, 15],
#     "D": [16, 17, 18],
#     "E": [19, 20, 21, 22, 23],
#     "F": [24, 25, 26, 27],
#     "G": [28, 29, 30, 31, 32],
#     "H": [33, 34, 35, 36],
#     "I": [37, 38, 39, 40, 41, 42],
#     "J": [43, 44, 45, 46, 47, 48, 49, 50]
# }

nodes_by_region = {
    "A": [1, 2, 3, 4, 5],
    "B": [6, 7, 8, 9],
    "C": [10, 11, 12],
    "D": [14]
}

n_sampling = 3  # 采样轮数
n_nodes = 7  # 每轮采样的节点数
regions = list(nodes_by_region.keys())  # 区域列表
selected_nodes_history = []  # 记录历史采样结果，用于目标2的计算   # 上一步的采样结果？？
# 在开始循环之前添加下面这行代码
selected_nodes_history_scores = []


for i in range(n_sampling):
    # 初始化采样结果
    selected_nodes = []
    selected_counts = {region: 0 for region in nodes_by_region}
    selected_nodes_with_counts = []

    # 遍历历史采样结果，统计各节点出现的次数
    for hist_nodes in selected_nodes_history:
        for node in hist_nodes:
            count = sum([node == n for n, _ in selected_nodes_with_counts])
            selected_nodes_with_counts.append((node, count)) # 统计在历史采样结果中，节点 node 出现的次数

    # 计算目标1的权重，根据上一轮采样结果的目标1和目标2的得分
    if len(selected_nodes_history) == 0:
        w1 = 1
    else:
        prev_score1, prev_score2 = selected_nodes_history_scores[-1]
        w1 = pow(1 - prev_score1, 2) + pow(1 - prev_score2, 2)


    nodes_by_region_copy = {r: nodes_by_region[r][:] for r in regions} # 创建一个区域节点的副本，以便在贪心算法中进行节点选择时，不对原始的区域节点列表产生影响。
    

    # # 按照目标1选择节点，每个区域至少选一个
    # for region in regions: # 循环每个区域
    #     if len(nodes_by_region[region]) > 0: # 如果该区域有节点
    #         node = random.choice(nodes_by_region[region]) # 随机选一个
    #         selected_nodes.append(node) # 加入选择列表
    #         selected_counts[region] += 1 # 区域选择计数+1
    #         nodes_by_region_copy[region].remove(node)

    n_nodes_left = n_nodes - len(selected_nodes) # 每个区域选择一个后，剩下的可选择次数


    # print("selected_counts:",selected_counts)
    # print("selected_nodes:", selected_nodes)
    # 使用贪心算法选择剩余的节点
    while n_nodes_left > 0:
        max_score = float('-inf')
        max_score_node = None
        max_score_region = None

        for region in regions: # 循环每个区域
            if len(nodes_by_region_copy[region]) > 0: # 如果区域中有节点
                for node in nodes_by_region_copy[region]: # 循环该区域的2个节点
                    # selected_counts字典: 键为区域，值为区域中选择节点的计数
                    '''
                    1. +1相当于多选一个node
                    2. sum 相当于统计当前node的选择次数
                    '''
                    # 计算该节点的得分
                    # score1，当前选择和中位数之间的差距，越小越好，差距越小说明达到需求，我们要选差距大的减少误差
                    # score2，进入该区域后，该节点被选择过的次数，越小说明没选过，我就要多选你
                    # score1 = abs(selected_counts[region] + 1 - sum(selected_counts.values()))  # 目标1的得分，不合理，在每一轮中，它对所有区域都是一样的
                    score1 = abs(selected_counts[region] + 1 - 4) # 4为每个区域的平均数目
                    score2 = sum([count for n, count in selected_nodes_with_counts if n == node])  # 目标2的得分，前面这个节点被选择过的次数
                    print(node, score1, score2, w1)
                    if len(selected_nodes_history) > 0:
                        score2 /= len(selected_nodes_history)

                    # 计算该节点的总得分
                    score = w1 * score1 + score2

                    if score > max_score:
                        max_score = score
                        max_score_node = node
                        max_score_region = region

        selected_nodes.append(max_score_node)
        # print("1.",selected_nodes)
        selected_counts[max_score_region] += 1

        nodes_by_region_copy[max_score_region].remove(max_score_node)
        n_nodes_left -= 1

    # 将采样结果记录到历史采样结果中
    selected_nodes_history.append(selected_nodes)

    # 记录目标1和目标2的得分
    total_count = sum(selected_counts.values())
    max_count = max(selected_counts.values())
    score1 = max_count - (total_count - max_count)
    score2 = sum([count / len(selected_nodes_history) for n, count in selected_nodes_with_counts])
    selected_nodes_history_scores.append((score1, score2))

    print("第{}轮采样结果:\n{}".format(i+1, selected_nodes))
    print("目标1得分: {}".format(score1))
    print("目标2得分: {}\n".format(score2))


'''
这段代码有以下几个特点：

目标1的权重根据历史采样结果动态计算。具体来说，第一轮采样时，设目标1的权重为1；之后每轮采样结束后，根据上一轮采样结果的目标1和目标2的得分计算权重，采用指数衰减算法，使得历史结果对当前结果的影响越来越小。
每轮先选取每个区域中的一个节点，保证每个区域至少有一个节点被选择。
剩余的节点采用贪心算法选择，对于每个节点，计算它在目标1和目标2下的得分，然后按总得分从大到小排序，选取得分最高的节点加入采样结果中。
目标2的计算借助了历史采样结果。具体来说，对于当前要采样的节点，遍历历史采样结果，计算它在历史采样结果中的出现次数，并将其除以历史采样结果的长度得到平均出现次数，作为该节点的目标2得分。
希望这次回答能够满足您的要求，如果还有问题，请随时提问！
'''