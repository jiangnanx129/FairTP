import os
import argparse
import numpy as np
import pandas as pd

# 数据标准化
class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    
'''
该函数主要用于生成时间序列数据的特征表示和索引。

df：输入的时间序列数据，是一个DataFrame。
x_offsets：输入数据的时间偏移量，表示使用过去数据的时间步长。
y_offsets：输入数据的时间偏移量，表示预测目标在未来的时间步长。
add_time_of_day：是否添加时间信息到特征中的布尔值参数。
add_day_of_week：是否添加星期几信息到特征中的布尔值参数。
函数首先将DataFrame中的数据转换为NumPy数组，并在最后一个维度上添加一个维度。然后，根据add_time_of_day和add_day_of_week参数的值，生成对应的时间特征表示。最后，将所有特征按照最后一个轴（维度）拼接成一个三维数组。同时，根据x_offsets和y_offsets计算出索引的最小值min_t和最大值max_t，并生成一个从min_t到max_t-1的整数数组作为索引。

函数返回生成的数据特征表示和索引。
'''
def generate_data_and_idx(df, x_offsets, y_offsets, add_time_of_day, add_day_of_week):
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)  # 将DataFrame（df）中的数据转换为NumPy数组，并在最后一个维度上添加一个维度
    '''
    original data shape: (35040, 716)
    df:                      1114091  1118333  ...  1123256  1122412
    Time                                   ...                  
    2019-01-01 00:00:00     29.0     23.0  ...     27.0     37.0
    2019-01-01 00:15:00     32.0     23.0  ...     26.0     40.0
    2019-01-01 00:30:00     40.0     24.0  ...     35.0     43.0
    2019-01-01 00:45:00     45.0     27.0  ...     27.0     46.0
    2019-01-01 01:00:00     90.0     49.0  ...     36.0     53.0
    ...                      ...      ...  ...      ...      ...
    2019-12-31 22:45:00     57.0     24.0  ...     33.0     43.0
    2019-12-31 23:00:00     59.0     29.0  ...     24.0     38.0
    2019-12-31 23:15:00     51.0     23.0  ...     24.0     39.0
    2019-12-31 23:30:00     43.0     18.0  ...     49.0     56.0
    2019-12-31 23:45:00     50.0     22.0  ...     22.0     37.0

    [35040 rows x 716 columns] # 和meta.csv对应的
    '''
    
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
    print("2. 拼接后数据维度：(n,716,3)", data.shape) # (105120, 716, 3)

    min_t = abs(min(x_offsets))  # x_offsets中的最小值的绝对值,11
    max_t = abs(num_samples - abs(max(y_offsets)))  # x_offsets中的最大值的绝对值和y_offsets中的最大值的绝对值之间的较小值，不包含边界（Exclusive）,xx-12
    print('idx min & max:', min_t, max_t) # 11 105108， 105120-12=105108
    idx = np.arange(min_t, max_t, 1)  # 生成一个从min_t到max_t-1的整数数组
    return data, idx  


# 生成四个文件，其中his.npz后面直接用！
def generate_train_val_test(args):
    years = args.years.split('_')
    df = pd.DataFrame()
    for y in years: # 拼接多年的数据
        df_tmp = pd.read_hdf("./data/"+args.dataset + '/' + args.dataset + '_his_' + y + '.h5')
        # df = df.append(df_tmp) # 原始代码报错，no attribute “append”
        df = pd.concat([df, df_tmp])
    print('original data shape:', df.shape)

    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y # 前12 预测 后12
    x_offsets = np.arange(-(seq_length_x - 1), 1, 1) # -11 to 1
    y_offsets = np.arange(1, (seq_length_y + 1), 1) # 1 to 13

    data, idx = generate_data_and_idx(df, x_offsets, y_offsets, args.tod, args.dow)
    # final data shape: (105120, 716, 3) idx shape: (105097,)
    print('final data shape:', data.shape, 'idx shape:', idx.shape) #idx为一维numpy，从11开始

    num_samples = len(idx)
    num_train = round(num_samples * 0.6)
    num_val = round(num_samples * 0.2)   

    # split idx
    idx_train = idx[:num_train]
    idx_val = idx[num_train: num_train + num_val]
    idx_test = idx[num_train + num_val:]
    
    # normalize
    x_train = data[:idx_val[0] - args.seq_length_x, :, 0] # 选取了除验证集之外的所有训练数据的第一个特征
    scaler = StandardScaler(mean=x_train.mean(), std=x_train.std())
    data[..., 0] = scaler.transform(data[..., 0]) # [..., 0]表示第一个特征维度，只针对第一个特征进行归一化，其他特征可能没有进行归一化处理。

    # save
    out_dir = args.dataset + '/' + args.years
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.savez_compressed(os.path.join(out_dir, 'his.npz'), data=data, mean=scaler.mean, std=scaler.std)

    np.save(os.path.join(out_dir, 'idx_train'), idx_train)
    np.save(os.path.join(out_dir, 'idx_val'), idx_val)
    np.save(os.path.join(out_dir, 'idx_test'), idx_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='sd', help='dataset name')
    parser.add_argument('--years', type=str, default='2019', help='if use data from multiple years, please use underline to separate them, e.g., 2018_2019')
    parser.add_argument('--seq_length_x', type=int, default=12, help='sequence Length')
    parser.add_argument('--seq_length_y', type=int, default=12, help='sequence Length')
    parser.add_argument('--tod', type=int, default=1, help='time of day')
    parser.add_argument('--dow', type=int, default=1, help='day of week')
    
    args = parser.parse_args()
    generate_train_val_test(args)
