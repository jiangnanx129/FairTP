import torch
import numpy as np
import sys
import os
import time
file_path = os.path.abspath(__file__) # /home/data/xjn/8fairness/src/engines/dcrnn_engine.py

from src.base.engine import BaseEngine # special
from src.utils.metrics import masked_mape, masked_rmse
from src.utils.metrics import compute_all_metrics
# 没有sample的操作，利用退火算法自适应采样
# 不同于4，只记录当前


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
    for district_id, nodes in district13_road_index.items():
        for node in nodes: # nodes为list, 循环938个节点！
            if node in sample_list:
                if district_id not in new_dict: # 返回sample_dict, 没有采样的区域就没有该区域id，0-12可能缺少7
                    new_dict[district_id] = []
                new_dict[district_id].append(node)

    # print(new_dict)
    return new_dict # 每个值一定是从小到大排列的！


def pro_sample(mape_for_each_region, condition_for_each_node, district13_road_index):
    node_cs = {}
    for v,node_list in district13_road_index.items():
        for node in node_list: # 循环该区域的节点列表：
            # node_cs[node] = 0.001*mape_for_each_region[int(v)] * condition_for_each_node[node]
            node_cs[node] = mape_for_each_region[int(v)] * condition_for_each_node[node]

    sorted_dict = dict(sorted(node_cs.items(), key=lambda x: x[0]))
    return sorted_dict

def resampel(sorted_dict,sam_num, district13_road_index):
    sample = []
    for v,node_list in district13_road_index.items():
        values_list = [sorted_dict[key] for key in node_list]
        max_key = node_list[values_list.index(max(values_list))]
        sample.append(max_key)
        
    a = list(sorted_dict.values())
    # a /= a.sum()
    min_value = min(a)
    max_value = max(a)

    normalized_list = [(x - min_value) / (max_value - min_value) for x in a]
    normalized_tensor = torch.tensor(normalized_list)
    keys_tensor = torch.tensor(list(sorted_dict.keys()))
    # 使用torch.multinomial()函数进行随机选择
    # sampled_indices = torch.multinomial(normalized_tensor, sam_num, replacement=False)
    mask = torch.ones_like(normalized_tensor, dtype=torch.bool)
    mask[sample] = False
    new_tensor = torch.masked_select(normalized_tensor, mask)
   
    sampled_indices = torch.multinomial(new_tensor, sam_num-len(sample), replacement=False)
    sampled_keys = keys_tensor[sampled_indices]
    sampled_list = sampled_keys.tolist()
    # sampled_list = np.random.choice(np.arange(len(list(sorted_dict.keys()))), size=sam_num, p=normalized_list)
    return sample+sampled_list


# 希望输出是每个区域一个(b,t,1)
def static_cal(pred,label): # (b, t, 13, 1), 13表示区域数量，1表示预测的交通值维度
    b, t = pred.shape[0], pred.shape[1]
    pred = pred.reshape(b * t, -1) # 将 pred 和 label 转换为形状为 (b*t, 13)
    label = label.reshape(b * t, -1)

    # 初始化一个张量用于保存每个区域对之间的 MAPE 差值
    mape_diff = [] # torch.zeros((78,))
    mape_for_each_region = [] # list长应该是13, 该区域和其他12个区域的mape差值的和

    # 计算 MAPE 差值
    idx = 0
    for i in range(pred.shape[1]-1): # 区域数量-1-->13-1=12 # i = 0123456789--11
        for j in range(i + 1, pred.shape[1]):
            mape_i = torch.abs(pred[:, i] - label[:, i]) / label[:, i] # 区域i的mape
            mape_j = torch.abs(pred[:, j] - label[:, j]) / label[:, j] # 区域j的mape
            # mape_diff[idx] = torch.mean(mape_i - mape_j)
            # mape_diff.append(torch.abs(torch.sum(mape_i - mape_j))) # mean换成sum 
            mape_diff.append(torch.sum(torch.abs(mape_i - mape_j)))
            idx += 1
        # print("--------------------:",mape_i, mape_j,mape_j.shape)

    region_mape_list = []
    
    for i in range(pred.shape[1]): # 循环13个区域
        mape_i = torch.abs(pred[:, i] - label[:, i]) / label[:, i] # 区域i的mape
        region_mape_list.append(mape_i) # 得到13个区域的mape
        mape_diff_for_region = 0
        for j in range(pred.shape[1]): # 循环13个区域
            if i==j:
                continue
            else:
                mape_j = torch.abs(pred[:, j] - label[:, j]) / label[:, j] # 区域j的mape,(b*n,)
                mape_diff_for_region += torch.sum(torch.abs(mape_i - mape_j))
                # mape_diff_for_region += torch.abs(torch.sum(mape_i - mape_j)) # 一个scalar
        mape_for_each_region.append(mape_diff_for_region/(pred.shape[1]-1))
        

    mape_diff_mean = torch.mean(torch.stack(mape_diff), dim=0)

    return mape_diff_mean, mape_for_each_region, region_mape_list # each为长为13的list 对应0-12个区域
      

def dynamic_cal(node_yl_dic2): # (b, t, 13, 1), 13表示区域数量，1表示预测的交通值维度
    differences = []
    node_keys = list(node_yl_dic2.keys())
    # 循环每个节点
    for i in range(len(node_keys)-1): 
        for j in range(i+1, len(node_keys)):
            sum_positive_i = sum(num for num in node_yl_dic2[node_keys[i]] if num > 0)
            sum_positive_j = sum(num for num in node_yl_dic2[node_keys[j]] if num > 0)
            sum_negtive_i = sum(num for num in node_yl_dic2[node_keys[i]] if num < 0)
            sum_negtive_j = sum(num for num in node_yl_dic2[node_keys[j]] if num < 0)

            diff = abs(sum_positive_i - sum_positive_j) + abs(sum_negtive_i - sum_negtive_j)
            # if type(diff) != torch.Tensor:
            #     print(node_yl_dic2[node_keys[i]], node_yl_dic2[node_keys[j]]) # 当2者都是tensor[0]的时候，后面5个数都是0（int）,stack报错！
            #     print("会死吗？", type(diff), diff, sum_positive_i, sum_positive_j, sum_negtive_i, sum_negtive_j)
            differences.append(diff)

    differences_mean = torch.mean(torch.stack(differences), dim=0)

    return differences_mean

# 比较1，提高效率，处理得到int0然后stack报错的情况
def dynamic_cal2(node_yl_dic2):
    node_keys = list(node_yl_dic2.keys())
    num_nodes = len(node_keys)

    sum_positives = torch.zeros(num_nodes, device='cuda:0')
    sum_negatives = torch.zeros(num_nodes, device='cuda:0')
    # sum_positives, sum_negatives = self._to_device(self._to_tensor([sum_positives, sum_negatives]))

    for i, nums in enumerate(node_yl_dic2.values()): # i节点对应下标，nums为list保存几个batch的受益/牺牲结果
        # nums_tensor = torch.stack(nums) # .to('cuda:1')
        # sum_positive = torch.max(nums_tensor, torch.zeros_like(nums_tensor)).sum(dim=0)
        # sum_negative = torch.min(nums_tensor, torch.zeros_like(nums_tensor)).sum(dim=0)
        # sum_positives[i] = sum_positive
        # sum_negatives[i] = sum_negative # 一定<=0

        for item in nums: # nums是list
            if item > 0:
                sum_positives[i] += item
            else: # item < 0:
                sum_negatives[i] += item


    differences = []
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            # if all(n == 0 for n in node_yl_dic2[node_keys[i]]) and all(n == 0 for n in node_yl_dic2[node_keys[j]]):
            #     diff = 0
            # else:
            diff = torch.abs(sum_positives[i] - sum_positives[j]) + torch.abs(sum_negatives[i] - sum_negatives[j])
            differences.append(diff)

    differences_mean = torch.mean(torch.tensor(differences))

    # if differences_mean ==0:
    #     print("===========:", node_yl_dic2) # , sum_positives,sum_negatives )
    return differences_mean

# 不考虑每个时刻，而是考虑3个时刻的总体情况, 没采到的节点设为0
def dynamic_cal3(node_yl_dic, district13_road_index): # 字典，键0-938，值一段时间该节点对应的受益/牺牲值
    condition_for_each_node = []
    sum_node_list = list(district13_road_index.values()) # sum_node_list所有节点
    flat_node_list = [num for sublist in sum_node_list for num in sublist] # 展开2维list

    new_node_yl_dic = dict(node_yl_dic) # node_yl_dic的修改不会影响new_node_yl_dic，它是独立的
    for node_938 in flat_node_list:
        if node_938 in new_node_yl_dic: # node_yl_dic中原本就有，该节点在过去采样中出现过
            continue
        else:
            new_node_yl_dic[node_938] = 0. # 键为0-937，值为对应的优劣情况
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sorted_new_node_yl_dic = {key: torch.tensor(new_node_yl_dic[key]).to(device) for key in sorted(new_node_yl_dic)} # sorted(new_node_yl_dic.items()) # 按照键从小到大排！
    
    all_num_nodes = len(list(sorted_new_node_yl_dic.keys()))
    for i in range(all_num_nodes):
        condition_i = sorted_new_node_yl_dic[i]
        condition_for_node = 0
        for j in range(all_num_nodes):
            if i==j:
                continue
            else:
                condition_j = sorted_new_node_yl_dic[j]
                # print((torch.abs(condition_i-condition_j)))
                condition_for_node += torch.abs(condition_i-condition_j)
        condition_for_each_node.append(condition_for_node/(all_num_nodes-1))
        

    node_keys = list(node_yl_dic.keys())
    num_nodes = len(node_keys) # 可能len为500(>450)

    differences = []
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            diff = torch.abs(node_yl_dic[node_keys[i]] - node_yl_dic[node_keys[j]])
            differences.append(diff)

    
    # differences_mean = torch.mean(torch.stack(differences), dim=0)
    differences_sum = torch.mean(torch.tensor(differences)) # sum

    return differences_sum, condition_for_each_node

# 输入就不是+1/-1，而是鉴别器输出, 鉴别器输出的倒数表示采样概率
def dynamic_cal4(dis_out, sample_map,node_yl_dic, district13_road_index): # 字典，键0-938，值一段时间该节点对应的受益/牺牲值
    sample_node_dict = {}
    condition_for_each_node = []
    dis_out_sum = torch.sum(1 / dis_out, dim=(0)) # (n,)
    for node in list(sample_map.keys()): # 0-449 # dis_out--(b*t,n), 
        sample_node_dict[sample_map[node]] = dis_out_sum[node]

    node_keys = list(node_yl_dic.keys())
    num_nodes = len(node_keys) # 可能len为500(>450)

    # 计算loss4，后面要加东西了
    differences = []
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            diff = torch.abs(node_yl_dic[node_keys[i]] - node_yl_dic[node_keys[j]])
            differences.append(diff)

    # differences_mean = torch.mean(torch.stack(differences), dim=0)
    differences_sum = torch.mean(torch.tensor(differences)) # sum



    sum_node_list = list(district13_road_index.values()) # sum_node_list所有节点
    flat_node_list = [num for sublist in sum_node_list for num in sublist] # 展开2维list
    for node in flat_node_list:
        if node not in list(sample_node_dict.keys()):
            sample_node_dict[node] = torch.tensor(2.0, requires_grad=True, device="cuda:1")  # 定义值为1/0.5的张量，并将其移到设备上

    # 938个节点和他们的受益/损失采样情况
    sorted_sample_node_dict = {k: v for k, v in sorted(sample_node_dict.items(), key=lambda item: item[0])}
    node_condition = []
    all_num_nodes = len(list(sorted_sample_node_dict.keys()))
    for i in range(all_num_nodes):
        condition_i = sorted_sample_node_dict[i]
        node_condition.append(condition_i)
        condition_for_node = 0
        for j in range(all_num_nodes):
            if i==j:
                continue
            else:
                condition_j = sorted_sample_node_dict[j]
                # print((torch.abs(condition_i-condition_j)))
                condition_for_node += torch.abs(condition_i-condition_j)
        condition_for_each_node.append(condition_for_node/(all_num_nodes-1))
        

    return differences_sum, condition_for_each_node, node_condition


def get_node_yl_dic(dis_labels, sample_map, district_nodes): # 这一个时刻的yl状况
    node_yl_dic = {}
    for v, node_list in district_nodes.items(): # 键0-12，值list,(0-449)
        for node in node_list: # (0-449)
            node_938 = sample_map[node] # node_938取值(0-937)
            if node_938 not in node_yl_dic:
                node_yl_dic[node_938] = 0
            node_yl_dic[node_938] += torch.sum(dis_labels[:,:, node, :].flatten()) # 应该是一个值，(n,1)select1-(1) # select_node[0]还select_node?????
    return  node_yl_dic     



class FSAMPE_Engine(BaseEngine):
    def __init__(self, criterion, T_dynamic, sample_num, district13_road_index, **args):
        super(FSAMPE_Engine, self).__init__(**args)
        self.criterion = criterion
        self.T_dynamic = T_dynamic
        self.sample_num = sample_num
        self.district13_road_index = district13_road_index


    def train_batch(self, sample_list, sample_map, sample_dict, district_nodes, yl_values, epoch):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        static_fair, dynamic_fair = [],[]
        self._dataloader['train_loader'].shuffle()
        batch_count = 0
        for X, label in self._dataloader['train_loader'].get_iterator(): # 循环每一个batch
            self._optimizer.zero_grad()

            X, label = self._to_device(self._to_tensor([X, label]))
           
            # 已划分好batch，先采样，再进model
            X, label= X[:, :, sample_list, :], label[:, :, sample_list, :] # (b,t,938,ci)--->(b,t,450,ci)
        
            pred, dis_labels, yl_label, dis_out, mape_data = self.model(X, label, sample_list, yl_values, self._iter_cnt)
            label = self._inverse_transform(label) # pred取值40+
            
            b,t,n,c = X.shape
            new_pred = torch.zeros(b, t, len(list(district_nodes.keys())), c)
            new_label = torch.zeros(b, t, len(list(district_nodes.keys())), c)

            for region, nodes_list in district_nodes.items(): # region是str '0' # sample2.py中所有round+0.5确保0-12均有采样，即有0-12
                new_pred[:, :, int(region)] = pred[:, :, nodes_list].mean(dim=2)#, keepdim=True)#.mean(dim=2)
                new_label[:, :, int(region)] = label[:, :, nodes_list].mean(dim=2)#, keepdim=True) #.mean(dim=2)
            
            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if new_label.min() < 1:
                mask_value = new_label.min()
            if self._iter_cnt == 0:
                print('check mask value', mask_value)

            loss1 = self._loss_fn(new_pred, new_label, mask_value) # 效用损失，区域的mae
            mape = masked_mape(new_pred, new_label, mask_value).item()
            rmse = masked_rmse(new_pred, new_label, mask_value).item()
            
            '''想办法给dis_out做反归一化！！！！！'''
            # dis_out.reshape(-1,2), yl_label.reshape(-1,).long() # softmax
            loss2 = self.criterion(dis_out.reshape(b,t,n,-1), yl_label.float()) # 鉴别器损失，450个节点的准确优劣分类
            loss3, mape_for_each_region, region_mape_list = static_cal(new_pred,new_label) # 静态公平正则化

            node_yl_dic = get_node_yl_dic(dis_labels, sample_map, district_nodes)
            loss4, condition_for_each_node, node_condition = dynamic_cal4(dis_out.reshape(b*t,-1), sample_map,node_yl_dic, self.district13_road_index)
            
            loss = loss1 + 40*loss2 + 0.1*loss3 + 0.15*loss4
            loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'
            self._logger.info(loss_message.format(epoch + 1, batch_count+1, loss1, loss2, loss3, loss4))

            sorted_dict = pro_sample(mape_for_each_region, condition_for_each_node, self.district13_road_index)
            sample_list = resampel(sorted_dict,self.sample_num,self.district13_road_index)
            sample_map = sum_map(sample_list, self.sample_num)
            sample_dict = sample_map_district(sample_list, self.district13_road_index)
            district_nodes = get_district_nodes(sample_dict, sample_map)
            # sample_list = resample(sorted_dict)

            loss.backward() # retain_graph=True)
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss1.item())
            train_mape.append(mape)
            train_rmse.append(rmse)
            static_fair.append(loss3.item())
            dynamic_fair.append(loss4.item())

            self._iter_cnt += 1
            batch_count += 1
        
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse), np.mean(static_fair), np.mean(dynamic_fair) # 一个epoch内！多个batch所以mean
    



    def train(self, sample_list, sample_map, sample_dict, district_nodes, yl_values):
        self._logger.info('Start training!') #

        wait = 0
        min_loss = np.inf
        for epoch in range(self._max_epochs):
            t1 = time.time()
            mtrain_loss, mtrain_mape, mtrain_rmse, mtrain_sfair, mtrain_dfair = self.train_batch(sample_list, sample_map, sample_dict, district_nodes, yl_values, epoch)
            t2 = time.time()

            v1 = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse, mvalid_sfair, mvalid_dfair = self.evaluate('val',sample_list, sample_map, sample_dict, district_nodes, yl_values, epoch)
            v2 = time.time()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            # 每个epoch，train一回，valid一回
            message = 'Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train SFair: {:.4f}, Train DFair: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Valid SFair: {:.4f}, Valid DFair: {:.4f}, Train Time: {:.4f}s/epoch, Valid Time: {:.4f}s, LR: {:.4e}'
            self._logger.info(message.format(epoch + 1, mtrain_loss, mtrain_rmse, mtrain_mape, mtrain_sfair, mtrain_dfair, \
                                             mvalid_loss, mvalid_rmse, mvalid_mape, mvalid_sfair, mvalid_dfair, \
                                             (t2 - t1), (v2 - v1), cur_lr))

            if mvalid_loss < min_loss:
                self.save_model(self._save_path)
                self._logger.info('Val loss decrease from {:.4f} to {:.4f}'.format(min_loss, mvalid_loss))
                min_loss = mvalid_loss
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info('Early stop at epoch {}, loss = {:.6f}'.format(epoch + 1, min_loss))
                    break
        
        self.evaluate('test',sample_list, sample_map, sample_dict, district_nodes, yl_values, epoch)



    # 直接用分层采样，看会不会效果更好. evaluate 和 train函数一样！！（模仿）
    def evaluate(self, mode, sample_list, sample_map, sample_dict, district_nodes, yl_values,epoch):
        if mode == 'test': # 如果是测试，加载模型
            self.load_model(self._save_path)
        self.model.eval()

        preds = []
        labels = []
        e_loss = []
        e_mape = []
        e_rmse = []
        estatic_fair, edynamic_fair = [],[]
        dis_labels_l, yl_label_l, dis_out_l, mape_data_l, dis_out_soft_l = [],[],[],[],[]
        dis_labels_list, sample_dict_list, sample_map_list, district_nodes_list, sample_list_list = [0 for _ in range(self.T_dynamic)], \
            [0 for _ in range(self.T_dynamic)],[0 for _ in range(self.T_dynamic)],[0 for _ in range(self.T_dynamic)],[0 for _ in range(self.T_dynamic)] # 先装入初始化的数据

        batch_count = 0
        with torch.no_grad():
            for X, label in self._dataloader[mode + '_loader'].get_iterator():
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                X, label= X[:, :, sample_list, :], label[:, :, sample_list, :] # (b,t,938,ci)--->(b,t,450,ci)
        
                pred, dis_labels, yl_label, dis_out, mape_data = self.model(X, label, sample_list, yl_values, self._iter_cnt)
                # pred, label = self._inverse_transform([pred, label])
                label = self._inverse_transform(label)

                b,t,n,c = X.shape
                new_pred = torch.zeros(b, t, len(list(district_nodes.keys())), c)
                new_label = torch.zeros(b, t, len(list(district_nodes.keys())), c)
                for region, nodes_list in district_nodes.items(): # region是str '0' # sample2.py中所有round+0.5确保0-12均有采样，即有0-12
                    new_pred[:, :, int(region)] = pred[:, :, nodes_list].mean(dim=2)#, keepdim=True)#.mean(dim=2)
                    new_label[:, :, int(region)] = label[:, :, nodes_list].mean(dim=2)#, keepdim=True) #.mean(dim=2)

                mask_value = torch.tensor(0)
                if new_label.min() < 1:
                    mask_value = new_label.min()

                loss1 = self._loss_fn(new_pred, new_label, mask_value) # 效用损失
                mape = masked_mape(new_pred, new_label, mask_value).item()
                rmse = masked_rmse(new_pred, new_label, mask_value).item()
                # dis_out.reshape(-1,2), yl_label.reshape(-1,).long() # softmax
                loss2 = self.criterion(dis_out.reshape(b,t,n,-1), yl_label.float())
                loss3, mape_for_each_region, region_mape_list = static_cal(new_pred,new_label) # 静态公平正则化

                node_yl_dic = get_node_yl_dic(dis_labels, sample_map, district_nodes)
                loss4, condition_for_each_node, node_condition = dynamic_cal4(dis_out.reshape(b*t,-1), sample_map,node_yl_dic, self.district13_road_index)

                loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'
                self._logger.info(loss_message.format(epoch + 1, batch_count+1, loss1, loss2, loss3, loss4))

                sorted_dict = pro_sample(mape_for_each_region, condition_for_each_node, self.district13_road_index)
                sample_list = resampel(sorted_dict,self.sample_num,self.district13_road_index)
                sample_map = sum_map(sample_list, self.sample_num)
                sample_dict = sample_map_district(sample_list, self.district13_road_index)
                district_nodes = get_district_nodes(sample_dict, sample_map)


                e_loss.append(loss1.item())
                e_mape.append(mape)
                e_rmse.append(rmse)
                estatic_fair.append(loss3.item())
                edynamic_fair.append(loss4.item())

                self._iter_cnt += 1
                batch_count += 1
                

        if mode == 'val':
            # mae = self._loss_fn(preds, labels, mask_value).item()
            # mape = masked_mape(preds, labels, mask_value).item()
            # rmse = masked_rmse(preds, labels, mask_value).item()
            mvalid_loss, mvalid_mape, mvalid_rmse, mvalid_sfair, mvalid_dfair = \
                np.mean(e_loss), np.mean(e_mape), np.mean(e_rmse), np.mean(estatic_fair), np.mean(edynamic_fair)
                
            return mvalid_loss, mvalid_mape, mvalid_rmse, mvalid_sfair, mvalid_dfair

        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []
            test_sfair, test_dfair = [],[]

            log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test SFair: {:.4f}, Test DFair: {:.4f}'
            self._logger.info(log.format(np.mean(e_loss), np.mean(e_rmse), np.mean(e_mape), np.mean(estatic_fair), np.mean(edynamic_fair)))
