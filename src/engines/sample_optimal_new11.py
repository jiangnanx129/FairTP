class DynamicSampling:
    def __init__(self, partition_dict, target1_weight=0.5, target2_weight=0.5):
        """
        :param partition_dict: 分区字典
        :param target1_weight: 目标1的权重
        :param target2_weight: 目标2的权重
        """
        self.partition_dict = partition_dict
        self.target1_weight = target1_weight
        self.target2_weight = target2_weight

        # 用于记录节点的累计次数
        self.node_count = {area_idx: [0] * len(node_list) for area_idx, node_list in partition_dict.items()}

        # 初始化节点选择概率为0
        self.node_prob = {area_idx: [0] * len(node_list) for area_idx, node_list in partition_dict.items()}

        # 初始化需要选择的节点数量
        self.total_samples = 30

        # 初始化已选择的节点集合
        self.selected_nodes = set() # 返回的选择集合

        # 从每个分区中选择一个节点
        for area_idx, node_list in self.partition_dict.items():
            node_idx = np.random.choice(range(len(node_list)))
            self.selected_nodes.add((area_idx, node_idx))
            self.node_count[area_idx][node_idx] += 1

        # # 确保每个区域至少有一个节点被选择
        # while not all([any([(i, j) in self.selected_nodes for j in range(len(node_list))]) for i, node_list in self.partition_dict.items()]):
        #     area_idx_list = list(self.partition_dict.keys())
        #     np.random.shuffle(area_idx_list)
        #     for area_idx in area_idx_list:
        #         node_list = self.partition_dict[area_idx]
        #         node_idx = np.random.choice(range(len(node_list)))
        #         if (area_idx, node_idx) not in self.selected_nodes:
        #             self.selected_nodes.add((area_idx, node_idx))
        #             self.node_count[area_idx][node_idx] += 1
        #             break

    def sample(self):
        while len(self.selected_nodes) < self.total_samples:
            # 计算节点选择概率
            for area_idx, node_count in self.node_count.items():
                node_list = self.partition_dict[area_idx]
                total_node_count = sum(node_count)

                # 计算目标1，即每个区域内待选节点数目的差值
                target1 = [(max(node_count) - node_count[node_idx]) / (len(node_list) - 1) for node_idx in range(len(node_list))]

                # 计算目标2，即节点的累计次数
                target2 = [node_count[node_idx] / total_node_count for node_idx in range(len(node_list))]

                # 综合考虑目标1和目标2，计算节点选择概率
                node_prob = [(self.target1_weight * t1 + self.target2_weight * t2) for t1, t2 in zip(target1, target2)]
                self.node_prob[area_idx] = node_prob

            # 根据概率选择节点，更新节点的累计次数
            for area_idx, node_list in self.partition_dict.items():
                node_idx = np.random.choice(range(len(node_list)), p=self.node_prob[area_idx])
                if (area_idx, node_idx) not in self.selected_nodes:
                    self.selected_nodes.add((area_idx, node_idx))
                    self.node_count[area_idx][node_idx] += 1

        return self.selected_nodes


# 定义分区字典
partition_dict = {
    '区域1': ['节点1', '节点2', '节点3'],
    '区域2': ['节点4', '节点5'],
    '区域3': ['节点6', '节点7', '节点8', '节点9']
}

# 创建 DynamicSampling 对象
dynamic_sampling = DynamicSampling(partition_dict, target1_weight=0.5, target2_weight=0.5)

# 进行采样
selected_nodes = dynamic_sampling.sample()

# 打印采样结果
for area_idx, node_idx in selected_nodes:
    area = list(partition_dict.keys())[area_idx]
    node = partition_dict[area][node_idx]
    print(f"选中 {node}，位于 {area}")
