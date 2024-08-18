'''
优化问题的解决，python，有50个节点属于10个区域，每个区域的节点数目不一样，有分区字典指示,键为区域，值为该区域包含的节点列表。
根据两个目标的最小化，每次从这10个区域中一共选出30个节点，该选择进行3次，每次都要两个目标最小化。
目标1是每个区域选择的节点数目的差额尽可能小，目标2是同一区域内每个节点被选择的累计次数尽可能相等（确保选择的多样性）
'''

'''
多目标优化，遗传算法求解
通过模拟遗传、突变、选择等机制，从一个初始的群体出发，经过多代的演化，逐步寻找到最优解或接近最优解的解决方案。
遗传算法的演化过程主要包括三个步骤：选择、交叉和变异。
1. 选择（Selection）：根据适应度函数对群体中的染色体进行评估，适应度较高的染色体有更高的概率被选中参与下一代的繁殖过程。这样可以倾向性地保留优秀的基因。
2. 交叉（Crossover）：选中的染色体按照一定的交叉方式进行基因的交换，生成新的染色体。交叉的目的是通过基因的重新组合，产生新的个体，以增加搜索空间。
3. 变异（Mutation）：在交叉后的染色体中，以一定的概率对基因进行突变操作，即改变个别基因的值。变异的目的是引入新的基因，增加搜索的多样性。
通过不断地迭代上述步骤，选择优秀的个体并进行繁殖、交叉和变异，最终达到逐步优化的目标。
遗传算法具有并行度高、全局搜索能力强、自适应调整能力好等特点，广泛应用于函数优化、组合优化、机器学习、人工智能等领域。
'''
import random
from collections import defaultdict
from sample import *

# 定义分区字典
partition = {'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9], 'C': [10, 11, 12, 13], 'D': [14, 15, 16], 'E': [17, 18, 19], 'F': [20, 21, 22], 'G': [23, 24, 25], 'H': [26, 27], 'I': [28, 29, 30, 31, 32], 'J': [33, 34, 35, 36, 37]}

# 总共选出的节点数
n = 30

# 目标函数1：每个区域选择的节点数目的差额尽可能小
def obj_func1(sample_dict, individual, node_count_global): 
    '''
    sample_dict: 采样的节点的分布，字典，键0-12，值0-937的list
    individual表示采样的列表，也是遗传中的一个个体
    node_count_global：统计全局采样状况的字典，键0-937，值为采样计数
    '''
    mean_num = int(len(individual)/len(list(sample_dict.keys()))) # 450/13=平均采样数目，每个区域一样的数目
    selected_nodes_per_region = []
    diff = 0 # 原始方差
    for district_id, nodes in sample_dict.items(): # 循环采样好的字典，值0-937
        selected_nodes_per_region.append(len(nodes)) # 每个区域采样的节点数目
        diff += (len(nodes) - mean_num)**2 # 节点数目的方差

        # 帮忙统计节点的采样次数，方便objective2的计算！
        for node in nodes: # nodes取值0-937
            if node not in node_count_global:
                node_count_global[node] = 0
            node_count_global[node] +=1 # 采样计数

    return diff, node_count_global


# 目标函数2：同一区域内每个节点被选择的累计次数尽可能相等
def obj_func2(node_count_global, nodes_by_region):
    '''
    node_count_global: 全局采样次数统计字典，键为0-937，值为对应节点采样的次数
    nodes_by_region: 所有节点的分区字典，键为区域0-12，值为list，每个区域对应的节点0-937
    '''
    diff = 0 # 原始
    ever_sampe_nodes = list(node_count_global.keys())
    # 针对每个区域看采样偏差
    for district_id, nodes in nodes_by_region.items():
        region_sampe_nodes = [] # 某一区域采样过的节点次数
        for node in nodes: # node是0-937
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

# 适应度函数：将两个目标函数的值加权求和
def fitness(individual, sample_dict, node_count_global, nodes_by_region):
    return obj_func1(sample_dict, individual, node_count_global) + obj_func2(node_count_global, nodes_by_region)


# 初始化种群
def init_population(n_pop, n, nodes_by_region): # n_pop为方案的个数，生成n_pop个方案，n为每个方案中的采样数目
    population = []
    counts = 0
    while counts < n_pop: # 要100个方案         
        individual = random.sample(range(0,938), n) # 从0-937中采样30个节点
        sample_dict = sample_map_district(individual, nodes_by_region) # 采样的节点元组分配到区域，返回字典，键为区域，值为对应的list保存采样的元素
        if len(list(sample_dict.keys())) == len(list(nodes_by_region.keys())): # 每个区域都有被采样到
            counts +=1 
            population.append(individual)

    return population # 保存了n_pop个方案的list


# 选择函数：使用锦标赛选择算法
def selection(population, tournament_size, node_count_global, nodes_by_region):
    '''
    population 是当前种群中所有个体的集合。
    tournament_size 是锦标赛的规模 (即比赛参与者的数量)。
    '''
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(population, tournament_size)
        winner = max(tournament, key=lambda ind: fitness(ind, node_count_global, nodes_by_region)) # ind代表锦标赛中的一个参与者，即种群中的一个个体。
        selected.append(winner)
        population.remove(winner)  # 移除已选择的个体
    return selected # 







# 交叉函数：使用单点交叉算法
def crossover(parent1, parent2):
    pos = random.randint(1, n-1)
    child1 = parent1[:pos] + parent2[pos:]
    child2 = parent2[:pos] + parent1[pos:]
    return child1, child2

# 变异函数：使用单点变异算法
def mutation(individual, mutation_rate):
    for i in range(n):
        if random.random() < mutation_rate:
            individual[i] = random.randint(1, 37)
    return individual

# 遗传算法主函数
def genetic_algorithm(n_pop, n_generation, tournament_size, crossover_rate, mutation_rate, nodes_by_region):
    population = init_population(n_pop, n, nodes_by_region)
    for i in range(n_generation):
        print(f'Generation {i+1}')
        next_population = []
        for j in range(n_pop):
            # 选择
            parent1, parent2 = random.sample(selection(population, tournament_size), 2)
            # 交叉
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]  # 使用切片复制父代个体
            # 变异
            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)
            next_population.append(child1)
            next_population.append(child2)
        # 更新种群
        population = next_population
    # 返回最优个体
    return max(population, key=fitness)

# 运行遗传算法
result = genetic_algorithm(n_pop=100, n_generation=100, tournament_size=5, crossover_rate=0.8, mutation_rate=0.02)
print(result)
