import pickle
import json
import numpy as np
import pandas as pd
import os
import copy
import folium
import math
import geopandas as gpd
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

current_path = os.getcwd() # 获取当前路径 /home/data/xjn/23largest_baseline/LargeST/data/sd
print("current path:", current_path)

# 读取已有的shp文件，查看分区情况
def get_district_read():
    # 读取 SHP 文件
    shp_path = "./data/CA_Counties/CA_Counties_TIGER2016.shp"     # 174*14, 174个disstrict划分
    data = gpd.read_file(shp_path) # GeoDataFrame
    # 查看数据
    print(data, type(data))
    print(data.head()) # 查看前5行
    print(data.info()) # 查看GeoDataFrame的基本信息，每列名称，数据类型和非空值的计数
    print(data.columns) # 将返回包含列名的列表
    print(data.shape) # 将返回一个元组，包含行数和列数 174*14
    # print(data.crs) # 将返回 CRS 对象，描述数据的投影和坐标系统信息。


# 生成数据，从df到npy
def generate_data_and_idx(df, x_offsets, y_offsets, add_time_of_day, add_day_of_week):
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)  # 将DataFrame（df）中的数据转换为NumPy数组，并在最后一个维度上添加一个维度
    
    return data # (n,716,1)

# 原始分区，62
def compare_road_district():
    # years = args.years.split('_')
    # df = pd.DataFrame()
    # for y in years: # 拼接多年的数据
    #     df_tmp = pd.read_hdf(args.dataset + '/' + args.dataset + '_his_' + y + '.h5')
    #     # df = df.append(df_tmp) # 原始代码报错，no attribute “append”
    #     df = pd.concat([df, df_tmp])
    # print('original data shape:', df.shape)
    # data = generate_data_and_idx(df) # (n,716,1-ci)
    
    # 读取sd_meta.csv文件，取lat和lon行
    ca_meta = pd.read_csv('./data/gla/gla_meta.csv')
    lat = ca_meta['Lat'].to_numpy()
    lon = ca_meta['Lng'].to_numpy()
    # print("sd数据集的lat以及lng列表长度：", lat, lon) # (716,) (716,)
    lat_lon = np.stack((lat, lon), axis=1)
    # print(lat_lon, lat_lon.shape) # (716,2), 32,-117

    # CA_Counties/CA_Counties_TIGER2016 ca-places-boundaries/CA_Places_TIGER2016
    shp_path = "./data/ca-places-boundaries/CA_Places_TIGER2016.shp" # 读取 SHP 文件
    district_data = gpd.read_file(shp_path) # GeoDataFrame
    
    geo_dict = {}
    district_dict = {int(i): 0 for i in range(district_data.shape[0])} # 字典，键为174个区，值为每个区域的节点计数
    # print("初始的行政区字典。174个：", district_dict)
    for k, road in enumerate(lat_lon): # 循环938条路
        geo_dict[k] = [road[0], road[1]]
        
        # print("a:",road['geometry'])
        # shape = Polygon(road['geometry']) # 待判断形状变为Polygon对象
        point = Point(road[1], road[0]) # 待判断的点，属于174中的那个区域
        # print(point)
        # 判断一个几何形状是否属于polygon
        for v, polygon_element in enumerate(district_data["geometry"]): # 循环174个district, 174全是polygon
            if polygon_element.contains(point):
                district_dict[int(v)] += 1
    print("路段map后的174行政区字典：", district_dict)
    non_zero_pairs = {key: value for key, value in district_dict.items() if value != 0} # 只留下174中有map路段的区域
    print("有路段的区域：",non_zero_pairs) # 字典，键为有区域编号(取值范围0-173)，值为对应区域的map的路段的数目
    with open("data/raw_nonzero_district.json", "w") as file: # 174个行政区中，每个map到的区对应的节点数目
        json.dump(non_zero_pairs, file) # 使用json.dump()函数将列表写入文件, 字典，62个区对应键
    with open("data/938geo_dict.json", "w") as file: # 174个行政区中，每个map到的区对应的节点数目
        json.dump(geo_dict, file) # 使用json.dump()函数将列表写入文件, 字典，62个区对应键


'''
1. count,58个区，18列， dbf文件为excel
'''
# get_district_read()
compare_road_district() # sd到coun，map为0