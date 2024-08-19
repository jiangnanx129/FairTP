import os
import argparse
import numpy as np
import pandas as pd
import pickle
import json
import random
'''
hk数据集
'''
# 包含数据采样 716采样350
# 35040 716 3/1
# 数据标准化
class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    
# 读取6个月的数据
def data_process():
    files = os.listdir("./data/hk/speed") # 获取speed文件夹下所有npy文件 # print(files) # ['202011speed.npy', '202010speed.npy', '202012speed.npy', '202103speed.npy', '202101speed.npy', '202102speed.npy']
    # files.sort() # print(files) # ['202010speed.npy', '202011speed.npy', '202012speed.npy', '202101speed.npy', '202102speed.npy', '202103speed.npy']
    files = ['202010speed.npy']  
    
    # 创建一个空的数组，用于存储拼接后的数据
    concatenated_data = []
    # 依次读取文件并拼接
    for file_name in files: 
        data_path = './data/hk/speed/' + file_name
        data = np.load(data_path)
        concatenated_data.append(data)
        print(file_name,data.shape)

    concatenated_array = np.concatenate(concatenated_data, axis=1)

    # print(concatenated_array.shape) # (938, 52416)
    data = concatenated_array.transpose() # 转置后 (n,938) n为一共多少条数据，n为节点数
    return data


def generate_data_and_idx(df, x_offsets, y_offsets, add_time_of_day, add_day_of_week):
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)  # 将DataFrame（df）中的数据转换为NumPy数组，并在最后一个维度上添加一个维度
    
    feature_list = [data]
    
    # 如果要添加时间信息
    if add_time_of_day:
        time_ind = (df.index.values - df.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')  # 计算每个时间戳相对于当天起始时间的小时数
        time_of_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))  # 在数据维度上复制、转置，生成形状为(num_samples, num_nodes, 1)的数组
        print("1. time_of_day: ", time_of_day, time_of_day.shape) # (105120, 716, 1)
        feature_list.append(time_of_day)  # 添加到特征列表中
    
    # 如果要添加星期几信息
    if add_day_of_week:
        dow = df.index.dayofweek  # 获取每个时间戳对应的星期几
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))  # 在数据维度上复制、转置，生成形状为(num_samples, num_nodes, 1)的数组，用来表示星期几的信息
        day_of_week = dow_tiled / 7  # 将星期几信息映射到[0, 1]的范围内
        print("2. day_of_week: ", day_of_week, day_of_week.shape) # (105120, 716, 1)
        feature_list.append(day_of_week)  # 添加到特征列表中

    print("2. 原始数据维度：(n,716,1)", data.shape) # (105120, 716, 1)
    data = np.concatenate(feature_list, axis=-1)  # 按照最后一个轴（维度）拼接特征列表中的数组，形状为(num_samples, num_nodes, num_features)
    '''只取10000个数据'''
    # data = data[:10000]
    # num_samples = 10000
    print("2. 拼接后数据维度：(n,716,3)", data.shape) # (105120, 716, 3)

    min_t = abs(min(x_offsets))  # x_offsets中的最小值的绝对值,11
    max_t = abs(num_samples - abs(max(y_offsets)))  # x_offsets中的最大值的绝对值和y_offsets中的最大值的绝对值之间的较小值，不包含边界（Exclusive）,xx-12
    print('idx min & max:', min_t, max_t) # 11 105108， 105120-12=105108
    idx = np.arange(min_t, max_t, 1)  # 生成一个从min_t到max_t-1的整数数组
    return data, idx  

# 没有时间元素，不要后两个维度
def generate_data_and_idx2(data, x_offsets, y_offsets):
    num_samples, num_nodes = data.shape
    data = np.expand_dims(data, axis=-1)  # 将DataFrame（df）中的数据转换为NumPy数组，并在最后一个维度上添加一个维度
    '''只取10000个数据'''
    # data = data[:10000]
    # num_samples = 10000

    print("2. 原始数据维度：(n,938,1)", data.shape) # (105120, 716, 1)

    min_t = abs(min(x_offsets))  # x_offsets中的最小值的绝对值,11
    max_t = abs(num_samples - abs(max(y_offsets)))  # x_offsets中的最大值的绝对值和y_offsets中的最大值的绝对值之间的较小值，不包含边界（Exclusive）,xx-12
    print('idx min & max:', min_t, max_t) # 11 105108， 105120-12=105108
    idx = np.arange(min_t, max_t, 1)  # 生成一个从min_t到max_t-1的整数数组
    return data, idx  


# 生成四个文件，其中his.npz后面直接用！
def generate_train_val_test(args):
    raw_data = data_process() # 处理数据，可拼接多个月的数据，(938,n)--转置--(n,938)

    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y # 前12 预测 后12
    x_offsets = np.arange(-(seq_length_x - 1), 1, 1) # -11 to 1
    y_offsets = np.arange(1, (seq_length_y + 1), 1) # 1 to 13

    data2, idx2 = generate_data_and_idx2(raw_data, x_offsets, y_offsets) # 不带时间的1维数据
    print('final data shape:', data2.shape, 'idx shape:', idx2.shape) #idx为一维numpy，从11开始

    num_samples = len(idx2)
    num_train = round(num_samples * 0.6)
    num_val = round(num_samples * 0.2)   

    # split idx
    idx_train = idx2[:num_train]
    idx_val = idx2[num_train: num_train + num_val]
    idx_test = idx2[num_train + num_val:]
    
    # normalize
    x_train = data2[:idx_val[0] - args.seq_length_x, :, 0] # 选取了除验证集之外的所有训练数据的第一个特征
    scaler = StandardScaler(mean=x_train.mean(), std=x_train.std())
    data2[..., 0] = scaler.transform(data2[..., 0]) # [..., 0]表示第一个特征维度，只针对第一个特征进行归一化，其他特征可能没有进行归一化处理。

    # save
    out_dir = './data/'+ args.dataset + '/' + args.years2 # hk/202010 hk2/202010
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.savez_compressed(os.path.join(out_dir, 'his_notime.npz'), data=data2, mean=scaler.mean, std=scaler.std) # 1维数据

    np.save(os.path.join(out_dir, 'idx_train'), idx_train)
    np.save(os.path.join(out_dir, 'idx_val'), idx_val)
    np.save(os.path.join(out_dir, 'idx_test'), idx_test)

    # for fair
    sample_list, sample_dict, sample_map = data_initial_sample(data2) # 要没有时间的数据
    adj_initial_sample('./data/hk/adj_938.npy', sample_list) # 
    print("sample_list:", sample_list)
    print("sample_dict:", sample_dict)
    print("sample_map:", sample_map)


# 初始分层采样, input(b,t,n,c)-(t,n,c), output(t,n',c)
def initial_sample(sd_district, sam_num):
    
    sum_node_list = list(sd_district.values()) # 双层list
    sum_node = sum(len(sublist) for sublist in sum_node_list) # 总结点数
    # 计算每个区域的节点数目占总节点数目的比例, region_nodes是list, float，region是(3,4)
    region_proportions = {region: len(region_nodes) / sum_node for region, region_nodes in sd_district.items()}
    # 根据比例计算每个区域应该抽取的节点数目
    region_samples = {region: round(proportion * sam_num + 0.5) for region, proportion in region_proportions.items()}
    
    zero_keys = [key for key, value in region_samples.items() if value == 0]
    count = len(zero_keys)
    print("1.初始采样为0的区域列表：", zero_keys)
    print("2.有x个区域采样为0：", count)
    print("3.所有区域的最小采样数：", min(list(region_samples.values())))

    print("=======================================================")
    print("1.初始总采样数目（还没开始）：", sum(list(region_samples.values())))
    print("2.各区域采样情况字典：", region_samples)
    print("==================== start sample =====================")

    sample = []
    for region, sample_count in region_samples.items():
        # 随机抽样，从几个节点中随机选！不适用于后面采样，只适用于初始化采样
        region_node_list = sd_district[region] # 某个区域的节点列表
        # 随机采样，后面可以换成切片！分数高的在前/后，直接切片
        nodes_sample = random.sample(region_node_list, sample_count)
        sample.extend(nodes_sample)
        # sample.extend(np.random.choice(nodes_by_region[region], size=sample_count, replace=False))
    print("3. 初步采样总结点数目（已初步采完）：", len(sample)) 

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
    print("=======================================================")
    print("最终采样结果：", sample)  
    print("最终采样数目：", len(sample)) 

    return sample # 采样的列表，每个元素为节点的index（0-937）


# 数据的初始化采样
def data_initial_sample(data):
    # 数据采样，一直不变，从到到尾都是那450个节点
    with open('./data/hk/district13_roadindex.json', 'r') as file: # 打开 JSON 文件
        district13_road_index = json.load(file) # 读取文件内容, 字典，键0-12，值为list(0-937)
    sample_list = initial_sample(district13_road_index, args.sam_num) # 返回采样的list，其中元素为节点index(0-938)
    sample_map = sum_map(sample_list, args.sam_num) # 返回采样匹配字典，键为450个节点（0-449），值为对应的节点（0-938）
    sample_dict = sample_map_district(sample_list, district13_road_index) # 返回采样分区结果字典，键为区域（0-12），值为该区域对应的节点（0-938）
    sample_data = data[:, sample_list, :] # 从716中取采样的350个节点
    print("采样后的数据shape:", sample_data.shape)
    # save
    out_dir1 = './data/'+args.dataset + '/' + args.years2
    if not os.path.exists(out_dir1):
        os.makedirs(out_dir1)
    np.savez_compressed(os.path.join(out_dir1, 'his_initial200.npz'), data=sample_data, sample_list=sample_list, sample_dict=sample_dict, sample_map=sample_map) #, mean=scaler.mean, std=scaler.std)

    return sample_list, sample_dict, sample_map # 采样的list，保存采样的节点的index(0-938)

def adj_initial_sample(adj_path, sample_list):
    # adj采样，一直不变，从到到尾都是那450个节点
    raw_adj = np.load(adj_path)
    new_adj = raw_adj[sample_list] # 8k*8k, select 716*8k
    new_adj = new_adj[:,sample_list]
    print("新矩阵：", new_adj.shape) # 返回 716*716 的邻接矩阵
    # save
    out_dir1 = './data/'+args.dataset + '/' + args.years2
    # np.save(os.path.join(out_dir1, 'adj_initial450'), new_adj) # 716
    # new_adj[new_adj != 0] = 1
    np.save(os.path.join(out_dir1, 'adj_initial200_all1'), new_adj)
    print("finish!")


def get_district_nodes(sample_dict, sample_map): # 2字典，sample_dict键0-12，值list(0-937); sample_map键0-449,值(0-938)
    district_nodes = {}
    for v, node_list in sample_dict.items(): # 一定有0-12，但是select_list可能为空：某个区域没有采样节点！会导致engine的new_pred报错(102行)
        # select_list = []
        # for key, value in sample_map.items():
        #     if value in node_list:
        #         select_list.append(key)
        # print("selecy_list:",select_list)

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
    for district_id, nodes in district13_road_index.items(): # 键值(3,4), 值list(0-715)
        for node in nodes: # nodes为list, 循环938个节点！
            if node in sample_list:
                if district_id not in new_dict: # 返回sample_dict, 没有采样的区域就没有该区域id，0-12可能缺少7
                    new_dict[district_id] = []
                new_dict[district_id].append(node)

    # print(new_dict)
    return new_dict # 每个值一定是从小到大排列的！   

# 为fairGNN得到优劣标签！
def data_initial_sample(data):
    # 数据采样，一直不变，从到到尾都是那450个节点
    with open('./data/hk/district13_roadindex.json', 'r') as file: # 打开 JSON 文件
        district13_road_index = json.load(file) # 读取文件内容, 字典，键0-12，值为list(0-937)
    sum_node_list = list(sd_district.values()) # 双层list
    sum_node = sum(len(sublist) for sublist in sum_node_list) # 总结点数
    # 计算每个区域的节点数目占总节点数目的比例, region_nodes是list, float，region是(3,4)
    region_proportions = {region: len(region_nodes) / sum_node for region, region_nodes in sd_district.items()}
    # 根据比例计算每个区域应该抽取的节点数目
    region_samples = {region: round(proportion * sam_num + 0.5) for region, proportion in region_proportions.items()}
    
    zero_keys = [key for key, value in region_samples.items() if value == 0]
    count = len(zero_keys)
    print("1.初始采样为0的区域列表：", zero_keys)
    print("2.有x个区域采样为0：", count)
    print("3.所有区域的最小采样数：", min(list(region_samples.values())))

    print("=======================================================")
    print("1.初始总采样数目（还没开始）：", sum(list(region_samples.values())))
    print("2.各区域采样情况字典：", region_samples)
    print("==================== start sample =====================")

    sample = []
    for region, sample_count in region_samples.items():
        # 随机抽样，从几个节点中随机选！不适用于后面采样，只适用于初始化采样
        region_node_list = sd_district[region] # 某个区域的节点列表
        # 随机采样，后面可以换成切片！分数高的在前/后，直接切片
        nodes_sample = random.sample(region_node_list, sample_count)
        sample.extend(nodes_sample)
        # sample.extend(np.random.choice(nodes_by_region[region], size=sample_count, replace=False))
    print("3. 初步采样总结点数目（已初步采完）：", len(sample)) 

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
    print("=======================================================")
    print("最终采样结果：", sample)  
    print("最终采样数目：", len(sample)) 

    return sample # 采样的列表，每个元素为节点的index（0-937）


    
    
    
    sample_list = initial_sample(district13_road_index, args.sam_num) # 返回采样的list，其中元素为节点index(0-938)
    sample_map = sum_map(sample_list, args.sam_num) # 返回采样匹配字典，键为450个节点（0-449），值为对应的节点（0-938）
    sample_dict = sample_map_district(sample_list, district13_road_index) # 返回采样分区结果字典，键为区域（0-12），值为该区域对应的节点（0-938）
    sample_data = data[:, sample_list, :] # 从716中取采样的350个节点
    print("采样后的数据shape:", sample_data.shape)
    out_dir1 = './data/'+args.dataset + '/' + args.years2
    if not os.path.exists(out_dir1):
        os.makedirs(out_dir1)
    np.savez_compressed(os.path.join(out_dir1, 'his_initial450.npz'), data=sample_data, sample_list=sample_list, sample_dict=sample_dict, sample_map=sample_map) #, mean=scaler.mean, std=scaler.std)

    return sample_list, sample_dict, sample_map # 采样的list，保存采样的节点的index(0-938)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hk2', help='dataset name')
    # parser.add_argument('--years', type=str, default='2019', help='if use data from multiple years, please use underline to separate them, e.g., 2018_2019')
    parser.add_argument('--years2', type=str, default='202010', help='if use data from multiple years, please use underline to separate them, e.g., 2018_2019')
    
    parser.add_argument('--seq_length_x', type=int, default=12, help='sequence Length')
    parser.add_argument('--seq_length_y', type=int, default=12, help='sequence Length')
    parser.add_argument('--tod', type=int, default=1, help='time of day')
    parser.add_argument('--dow', type=int, default=1, help='day of week')
    parser.add_argument('--sam_num', type=int, default='200', help='sample sum')
    
    
    args = parser.parse_args()
    generate_train_val_test(args)
