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
from shapely.ops import unary_union
from folium.plugins import HeatMap
from folium.features import GeoJson
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

current_path = os.getcwd() # 获取当前路径 /home/data/xjn/23largest_baseline/LargeST/data/sd
print("current path:", current_path)

'''
画格子分区
'''

# 生成数据，从df到npy
def generate_data_and_idx(df, x_offsets, y_offsets, add_time_of_day, add_day_of_week):
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)  # 将DataFrame（df）中的数据转换为NumPy数组，并在最后一个维度上添加一个维度
    
    return data # (n,716,1)

# 原始分区，62
def get_lat_lon(data_path):
    # years = args.years.split('_')
    # df = pd.DataFrame()
    # for y in years: # 拼接多年的数据
    #     df_tmp = pd.read_hdf(args.dataset + '/' + args.dataset + '_his_' + y + '.h5')
    #     # df = df.append(df_tmp) # 原始代码报错，no attribute “append”
    #     df = pd.concat([df, df_tmp])
    # print('original data shape:', df.shape)
    # data = generate_data_and_idx(df) # (n,716,1-ci)
    
    # 读取sd_meta.csv文件，取lat和lon行
    ca_meta = pd.read_csv(data_path+"gba_meta.csv")
    lat = ca_meta['Lat'].to_numpy()
    lon = ca_meta['Lng'].to_numpy()
    lat_lon = np.stack((lat, lon), axis=1)
    max_lat, min_lat, max_lon, min_lon = max(lat),min(lat),max(lon),min(lon)
    
    print("gba的区域范围：", max_lat, min_lat, max_lon, min_lon)
    # 35.602653 33.404943 -114.559672 -119.434005
    return lat_lon, max_lat, min_lat, max_lon, min_lon


def partition(lat_lon, max_lon, min_lon, max_lat, min_lat):
    # 计算每个区域的纬度和经度间隔
    latitude_interval = 0.4 #  (max_lat - min_lat) / x1
    longitude_interval = 0.4 # (max_lon - min_lon) / x1
    la_num = round((max_lat-min_lat)/latitude_interval + 0.5) # 向上取整
    lo_num = round((max_lon-min_lon)/longitude_interval + 0.5) # 向上取整
    district_dict = {} # 键值为lsit，对应区域的节点index(0-3833)  # {int(i): 0 for i in range(la_num*lo_num)} # 字典，键为174个区，值为每个区域的节点计数
    district_counts = {} # 键(3,4)表示区域，值为对应区域的节点数目
    district_geo = {} # 键同上，值为对应区域的范围！

    district_numpy = np.zeros((la_num, lo_num))
    # 遍历节点坐标并统计每个区域的节点数目
    for v, coordinate in enumerate(lat_lon):
        lat, lon = coordinate[0], coordinate[1]

        # 计算区域索引
        lat_index = math.floor((lat - min_lat) / latitude_interval)
        lon_index = math.floor((lon - min_lon) / longitude_interval)
        # # 确保索引在合法范围内
        # lat_index = min(lat_index, x1-1)
        # lon_index = min(lon_index, x1-1)
        district_numpy[lat_index][lon_index] += 1 # 对应区域的节点数目加一
        if (lat_index, lon_index) not in district_dict:
            district_dict[(lat_index, lon_index)] = []
        district_dict[(lat_index, lon_index)].append(v) # 键(3,4)表示区域，值为列表，各区域对应节点index(0-3833)
        
        if (lat_index, lon_index) not in district_geo:
            l_lat = min_lat + lat_index*latitude_interval
            r_lat = min_lat + (lat_index+1) *latitude_interval
            l_lon = min_lon + lon_index*latitude_interval
            r_lon = min_lon + (lon_index+1) *latitude_interval
            district_geo[(lat_index, lon_index)] = [l_lat,r_lat,l_lon,r_lon]

    for v, (key, values) in enumerate(district_dict.items()):
        district_counts[v] = len(values)


    # print("节点总数，=3834表示全部map：", district_numpy, np.sum(district_numpy)) 
    # print("节点分布字典：", district_dict)
    # print("区域中节点数目字典：", district_counts)
    # print("区域位置字典：", district_geo)
    with open(data_path+"gba_district.json", 'wb') as file:
        pickle.dump(district_dict, file)
    with open(data_path+"gba_district_counts.json", 'wb') as file:
        pickle.dump(district_counts, file)
    with open(data_path+"gba_district_geo.json", 'wb') as file:
        pickle.dump(district_geo, file)
    # with open(data_path+"gla_district.json", "w") as file: # 174个行政区中，每个map到的区对应的节点数目
    #     json.dump(district_dict, file) # 使用json.dump()函数将列表写入文件, 字典，62个区对应键
    # with open(data_path+"gla_district_geo.json", "w") as file: # 174个行政区中，每个map到的区对应的节点数目
    #     json.dump(district_geo, file) # 使用json.dump()函数将列表写入文件, 字典，62个区对应键
    

    '''
    gla的区域范围： 35.602653 33.404943 -114.559672 -119.434005
    [[  0.   0. 428. 859. 143.  16.  24.   4.   5.   6.]
    [ 14. 382. 946. 544. 337.  23.   0.   0.   0.   0.]
    [  0.  21.  10.   0.  28.   0.   0.   0.   0.   0.]
    [  0.   0.   0.   0.   2.  10.  14.   4.   0.   0.]
    [  0.   0.   0.   0.   0.   0.   0.  10.   4.   0.]] 3834.0
    '''

def district_heatmap4(hb_regions_polygon, hb_regions_counts):
    polygons = list(hb_regions_polygon.values())
    weights = list(hb_regions_counts.values())
    df = pd.DataFrame({
        'geometry': polygons,
        'weight': weights
    })
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    fig, ax = plt.subplots(1, 1)
    # 绘制热力图，使用'YlOrRd'颜色映射
    gdf.plot(column='weight', cmap='YlOrRd', linewidth=0.8, ax=ax, edgecolor='0.8')

    # 在每个polygon的中心位置添加权重文本
    for x, y, label in zip(gdf.geometry.centroid.x, gdf.geometry.centroid.y, gdf["weight"]):
        ax.text(x, y, label, fontsize=7)

    # 添加色条（colorbar）
    norm = plt.Normalize(vmin=min(weights), vmax=max(weights))
    sm = cm.ScalarMappable(cmap='YlOrRd', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', label='Weight')
    # cbar.ax.locator_params(nbins=5)  # 控制色条刻度数量
    # cbar.ax.yaxis.set_ticks_position('left')
    # cbar.ax.yaxis.set_label_position('left')
    # cbar.ax.yaxis.set_ticks([min(weights), max(weights)])  # 控制色条刻度位置
    # cbar.ax.yaxis.set_ticklabels(['Min', 'Max'])  # 控制色条刻度标签

    # # 调整色条长度
    # cbar.ax.set_aspect(20)

    # 保存图像
    plt.savefig("heatmap4.png", dpi=300)
    plt.show()


# 13个区域中的节点数目统计图
def plot_roadcount_statistics(hb_regions_counts): # data是字典
    # 将字典转换为 Pandas 的 DataFrame
    df = pd.DataFrame.from_dict(hb_regions_counts, orient='index', columns=['Road Count'])
    print(df)
    # 按人口数量降序排序
    # df = df.sort_values(by='Population', ascending=False)

    # 创建一个条形图
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.bar(df.index, df['Road Count'], color='steelblue')

    # 设置标题和轴标签
    ax.set_title('Road Counts Statistics of Hong Kong Districts')
    ax.set_xlabel('District')
    ax.set_ylabel('Road Counts')

    # 添加数值标签
    for i, v in enumerate(df['Road Count']):
        ax.text(i, v + 10000, str(v), ha='center', va='bottom')

    # 调整横轴标签的显示
    plt.xticks(rotation=30, ha='right')

    # 调整图像布局
    plt.tight_layout()
    plt.savefig("Barchart_13_count_gba.png", dpi=300)
    # 显示图像
    plt.show()





'''
1. gba: 1.930537000000001 1.5270620000000008
2. 2352节点！
'''
data_path = './data/gba/'
lat_lon, max_lat, min_lat, max_lon, min_lon = get_lat_lon(data_path)
partition(lat_lon, max_lon, min_lon, max_lat, min_lat)
print(max_lat-min_lat, max_lon-min_lon)
with open(data_path+"gba_district_counts.json", 'rb') as file:
    hb_regions_counts = pickle.load(file)
plot_roadcount_statistics(hb_regions_counts)

