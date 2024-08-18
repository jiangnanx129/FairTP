import torch
import numpy as np
import sys
import os
import time
file_path = os.path.abspath(__file__) # /home/data/xjn/8fairness/src/engines/dcrnn_engine.py

from src.base.engine import BaseEngine # special
from src.utils.metrics import masked_mape, masked_rmse
from src.utils.metrics import compute_all_metrics
from sample2 import * # 改进版本sample，适配

'''
# fairness sample train batch (ours)
'''
def static_cal(pred,label): # (b, t, 13, 1), 13表示区域数量，1表示预测的交通值维度
    b, t = pred.shape[0], pred.shape[1]
    pred = pred.reshape(b * t, -1) # 将 pred 和 label 转换为形状为 (b*t, 13)
    label = label.reshape(b * t, -1)

    # 初始化一个张量用于保存每个区域对之间的 MAPE 差值
    mape_diff = [] # torch.zeros((78,))

    # 计算 MAPE 差值
    idx = 0
    for i in range(pred.shape[1]-1): # 区域数量-1-->13-1=12
        for j in range(i + 1, pred.shape[1]):
            mape_i = torch.abs(pred[:, i] - label[:, i]) / label[:, i] # 区域i的mape
            mape_j = torch.abs(pred[:, j] - label[:, j]) / label[:, j] # 区域j的mape
            # mape_diff[idx] = torch.mean(mape_i - mape_j)
            mape_diff.append(torch.abs(torch.sum(mape_i - mape_j))) # mean换成sum
            idx += 1
        # print("--------------------:",mape_i, mape_j,mape_j.shape)

    mape_diff_mean = torch.mean(torch.stack(mape_diff), dim=0)
    return mape_diff_mean
      
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

    sum_positives = torch.zeros(num_nodes) #, device='cuda:1')
    sum_negatives = torch.zeros(num_nodes) #, device='cuda:1')

    for i, nums in enumerate(node_yl_dic2.values()): # i节点对应下标，nums为list保存几个batch的受益/牺牲结果
        nums_tensor = torch.stack(nums) # .to('cuda:1')
        sum_positive = torch.max(nums_tensor, torch.zeros_like(nums_tensor)).sum(dim=0)
        sum_negative = torch.min(nums_tensor, torch.zeros_like(nums_tensor)).sum(dim=0)
        sum_positives[i] = sum_positive
        sum_negatives[i] = sum_negative # 一定<=0

    differences = []
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            # if all(n == 0 for n in node_yl_dic2[node_keys[i]]) and all(n == 0 for n in node_yl_dic2[node_keys[j]]):
            #     diff = 0
            # else:
            diff = torch.abs(sum_positives[i] - sum_positives[j]) + torch.abs(sum_negatives[i] - sum_negatives[j])
            differences.append(diff)

    differences_mean = torch.mean(torch.tensor(differences))

    return differences_mean



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
        dis_labels_list, sample_dict_list, sample_map_list, district_nodes_list = [0 for _ in range(self.T_dynamic)],\
            [0 for _ in range(self.T_dynamic)],[0 for _ in range(self.T_dynamic)],[0 for _ in range(self.T_dynamic)] # 先装入初始化的数据

        for X, label in self._dataloader['train_loader'].get_iterator(): # 循环每一个batch
            self._optimizer.zero_grad()

            X, label = self._to_device(self._to_tensor([X, label]))
           
            # 已划分好batch，先采样，再进model
            X, label= X[:, :, sample_list, :], label[:, :, sample_list, :] # (b,t,938,ci)--->(b,t,450,ci)
        
            pred, dis_labels, yl_label, dis_out = self.model(X, label, sample_list, yl_values, self._iter_cnt)
            pred, label = self._inverse_transform([pred, label]) # pred取值40+
            # print("1:", pred.shape, dis_labels.shape, yl_label.shape, dis_out.shape) # 1: torch.Size([64, 12, 450, 1]) torch.Size([345600, 1]) torch.Size([64, 12, 450, 1]) torch.Size([345600, 1])

            # 计算损失：效用损失loss1, 对抗损失loss2，静态公平正则化loss3，动态公平正则化loss4（几个batch后才用到）
            
            # 此时pred，label均为（b,t,n,1）, 接下来，先分区，938分到13个行政区，在评估区域的预测精度！放在metrics中
            # baseline 不用改变分布采样！只要让结果从450个节点 到13个区域，(b,t,450,c)-(b,t,13,c)
            b,t,n,c = X.shape
            new_pred = torch.zeros(b, t, len(list(district_nodes.keys())), c)
            new_label = torch.zeros(b, t, len(list(district_nodes.keys())), c)

            if len(list(district_nodes.keys()))!=13:
                print("不是每个区域都有采样！")
                print("district_nodes: ",district_nodes)
                print("sample_map: ",sample_map)
            # print(new_pred.shape, pred.shape) # torch.Size([64, 12, 13, 1]) torch.Size([64, 12, 450, 1])
            # print("====:",district_nodes)
            for region, nodes_list in district_nodes.items(): # region是str '0' # sample2.py中所有round+0.5确保0-12均有采样，即有0-12
                new_pred[:, :, int(region)] = pred[:, :, nodes_list].mean(dim=2)#, keepdim=True)#.mean(dim=2)
                new_label[:, :, int(region)] = label[:, :, nodes_list].mean(dim=2)#, keepdim=True) #.mean(dim=2)
            # print("2:", new_pred.shape, new_label.shape) # 2: torch.Size([64, 12, 13, 1]) torch.Size([64, 12, 13, 1])
            
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
            loss2 = self.criterion(dis_out.reshape(b,t,n,-1), yl_label.float()) # 鉴别器损失，450个节点的准确优劣分类
            loss3 = static_cal(new_pred,new_label) # 静态公平正则化
            # self._logger.info("1:",dis_out.reshape(b,t,n,-1), yl_label.float()) # 基本全是-1

            if self._iter_cnt<3:
                if_loss4 = 0 # 没有动态
                # 重新采样，静态。 dis_labels（t*b*n,co）, 更新了sample_list, sample_map, sample_dict, district_nodes
                sample_list, sample_map, sample_dict, district_nodes = \
                    static_sample3(dis_labels.reshape(b,t,n,-1), sample_dict, sample_map, district_nodes, self.district13_road_index, self.sample_num)
                loss = loss1 + loss2 + loss3
                # pirnt()
                loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}'
                self._logger.info(loss_message.format(epoch + 1, self._iter_cnt+1, loss1, loss2, loss3))

            if self._iter_cnt >= self.T_dynamic: # 有动态公平, 动态公平采样
                if_loss4 = 1
                sample_list, sample_map, sample_dict, district_nodes, node_yl_dic, node_yl_dic2 = \
                    dynamic_sample3(dis_labels_list, sample_dict_list, sample_map_list, district_nodes_list, self.district13_road_index, self.sample_num)
                loss4 = dynamic_cal2(node_yl_dic2)
                loss = loss1 + loss2 + loss3 + loss4

                loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'
                self._logger.info(loss_message.format(epoch + 1, self._iter_cnt+1, loss1, loss2, loss3, loss4))

            dis_labels_list[self._iter_cnt%3], sample_dict_list[self._iter_cnt%3], sample_map_list[self._iter_cnt%3], \
                district_nodes_list[self._iter_cnt%3] = dis_labels.reshape(b,t,n,-1), sample_dict, sample_map, district_nodes
            
            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss1.item())
            train_mape.append(mape)
            train_rmse.append(rmse)
            static_fair.append(loss3.item())
            if if_loss4:
                dynamic_fair.append(loss4.item())

            self._iter_cnt += 1

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
            mvalid_loss, mvalid_mape, mvalid_rmse, mvalid_sfair, mvalid_dfair = self.evaluate('val',sample_list, sample_map, sample_dict, district_nodes, yl_values)
            v2 = time.time()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            # 每个epoch，train一回，valid一回
            message = 'Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train SFair: {:.4f}, Train DFair: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Valid SFair: {:.4f}, Valid DFair: {:.4f}, Train Time: {:.4f}s/epoch, Valid Time: {:.4f}s, LR: {:.4e}'
            self._logger.info(message.format(epoch + 1, mtrain_loss, mtrain_rmse, mtrain_mape, mtrain_sfair, mtrain_dfair, \
                                             mvalid_loss, mvalid_rmse, mvalid_mape, mvalid_sfair, mvalid_dfair \
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

        self.evaluate('test',sample_list, sample_map, sample_dict, district_nodes, yl_values)

    # 直接用分层采样，看会不会效果更好
    def evaluate(self, mode, sample_list, sample_map, sample_dict, district_nodes, yl_values):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()

        preds = []
        labels = []
        with torch.no_grad():
            for X, label in self._dataloader[mode + '_loader'].get_iterator():
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                X, label= X[:, :, sample_list, :], label[:, :, sample_list, :] # (b,t,938,ci)--->(b,t,450,ci)
        
                pred, dis_labels, yl_label, dis_out = self.model(X, label, sample_list, yl_values, self._iter_cnt)
                pred, label = self._inverse_transform([pred, label])

                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        # handle the precision issue when performing inverse transform to label
        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == 'val':
            mae = self._loss_fn(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return mae, mape, rmse

        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []
            print('Check mask value', mask_value)
            for i in range(self.model.horizon):
                res = compute_all_metrics(preds[:,i,:], labels[:,i,:], mask_value)
                log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])

            log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)))