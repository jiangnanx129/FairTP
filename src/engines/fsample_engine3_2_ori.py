import torch
import numpy as np
import sys
import os
import time
file_path = os.path.abspath(__file__) # /home/data/xjn/8fairness/src/engines/dcrnn_engine.py

from src.base.engine import BaseEngine # special
from src.utils.metrics import masked_mape, masked_rmse
from src.utils.metrics import compute_all_metrics
import random
from src.engines.sample import *
# from src.engines.sample_optimal_xhbl import * # 循环遍历实现优化采样
# from src.engines.sample_optimal_greedy_test import * # greedy实现优化采样，同时区域+节点
from src.engines.sample_optimal_greedy_test2 import * # greedy, 先区域后节点
'''
10月份新讨论，未加动静态约束于采样，将采样视为优化问题
1. 缓解数据不平衡  2. 确保采样的多样性
分开了loss和loss2
没有统计受益和牺牲的比例
'''

class FSAMPE_Engine(BaseEngine):
    def __init__(self, criterion, d_lrate, n_number,model_params, D_params, optimizer_D, scheduler_D, a_loss3, a_loss4, T_dynamic, sample_num, district13_road_index, **args):
        super(FSAMPE_Engine, self).__init__(**args)
        self.criterion = criterion
        self.T_dynamic = T_dynamic
        self.sample_num = sample_num
        self.district13_road_index = district13_road_index
        self.d_lrate = d_lrate
        self.n_number = n_number # 从n_number个组合中选出最合适的采样
        self.model_params = model_params # encoder+decoder的参数
        self.D_params = D_params # discriminator的参数
        self.optimizer_D = optimizer_D # BaseEngine有self._lr_scheduler = scheduler
        self.scheduler_D = scheduler_D # BaseEngine有self._lr_scheduler = scheduler
        self.a_loss3 = a_loss3 # 静态损失超参数
        self.a_loss4 = a_loss4

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = 'final_model_s{}_HK3_disout6_2b.pt'.format(self._seed)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))
    
    def load_model(self, save_path):
        filename = 'final_model_s{}_HK3_disout6_2b.pt'.format(self._seed)
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename)))
        
    # 单个epoch
    def train_batch(self, sample_list, sample_map, sample_dict, district_nodes, yl_values, epoch, total_nodes):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        dis_loss, static_fair, dynamic_fair =[],[],[]
        # self._dataloader['train_loader'].shuffle()
        batch_count = 0
        node_count_global = {} # 全局，贯穿1个epoch内所有batch记录采样过的个体情况，采样次数
        node_count_global = calcute_global_dic(node_count_global, sample_list) # greedy专用

        # dis_out_list, sample_dict_list, sample_map_list, district_nodes_list, sample_list_list = [0 for _ in range(self.T_dynamic)],\
        #     [0 for _ in range(self.T_dynamic)],[0 for _ in range(self.T_dynamic)],[0 for _ in range(self.T_dynamic)], [0 for _ in range(self.T_dynamic)] # 先装入初始化的数据
        for X, label in self._dataloader['train_loader'].get_iterator(): # 循环每一个batch
            # print(batch_count, sample_list)
            # if batch_count == 2:
            #     break
            '''
            co-train的两个loss
            参考：https://blog.csdn.net/qq_40737596/article/details/127674436 和 https://zhuanlan.zhihu.com/p/596917587
            1. loss.backward(retain_graph=True)，不释放中间变量的梯度，是鉴别器的输入
            2. 除了1，再加上在fsample3.py文件中，给中间变量加上.detach()： dcgru4_state_tensor.reshape(t,b,n,-1).detach()
            '''
            X, label = self._to_device(self._to_tensor([X, label]))
            X, label= X[:, :, sample_list, :], label[:, :, sample_list, :] # (b,t,938,ci)--->(b,t,450,ci)
            b,t,n,c = X.shape # n是采样的数目


            '''训练普通模型(除鉴别器), 先有鉴别器，基于鉴别器的结果，才能计算loss4'''
            self.model.classif.requires_grad_(False)
            self.model.encoder.requires_grad_(True)
            self.model.decoder.requires_grad_(True)
            self._optimizer.zero_grad()
           
            pred, dis_labels, yl_label, dis_out, mape_data = self.model(X, label, sample_list, yl_values, self._iter_cnt)
            # print("55. requires_grad:", pred.requires_grad, dis_labels.requires_grad, yl_label.requires_grad, dis_out.requires_grad, dcgru4_state_tensor.requires_grad)
            label1 = self._inverse_transform(label) # pred取值40+
            
            new_pred = torch.zeros(b, t, len(list(district_nodes.keys())), c)
            new_label = torch.zeros(b, t, len(list(district_nodes.keys())), c)
            for region, nodes_list in district_nodes.items(): # region是str '0' # sample2.py中所有round+0.5确保0-12均有采样，即有0-12
                new_pred[:, :, int(region)] = pred[:, :, nodes_list].mean(dim=2)#, keepdim=True)#.mean(dim=2)
                new_label[:, :, int(region)] = label1[:, :, nodes_list].mean(dim=2)#, keepdim=True) #.mean(dim=2)
            
            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if new_label.min() < 1:
                mask_value = new_label.min()
            if self._iter_cnt == 0:
                print('check mask value', mask_value)

            loss1 = self._loss_fn(new_pred, new_label, mask_value) # 效用损失，区域的mae
            mape = masked_mape(new_pred, new_label, mask_value).item()
            rmse = masked_rmse(new_pred, new_label, mask_value).item()
            
            loss2_1 = self.criterion(dis_out.reshape(b,t,n,-1), yl_label.float()) # 鉴别器损失，450个节点的准确优劣分类
            loss3, values_region = static_cal(new_pred, new_label) # 静态公平正则化
            loss4 = dynamic_cal5_global(dis_labels, dis_out, sample_map, district_nodes) # greedy
            # loss4, node_count_global = dynamic_cal5_global(dis_labels, dis_out, sample_map, district_nodes, self.district13_road_index, node_count_global) # xhbl更新了node_count_global
            '''加上detach后，loss4没有梯度'''
            # print("66. requires_grad:", loss1.requires_grad, loss2_1.requires_grad, loss3.requires_grad, loss4.requires_grad) # TTTT

            loss = loss1 + self.a_loss3*loss3 + self.a_loss4*loss4  # HK 0.1, dis_lobel 0.01
            loss.backward()#retain_graph=True)
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model_params, self._clip_grad_value)
            self._optimizer.step()

            self.model.encoder.requires_grad_(False)
            self.model.decoder.requires_grad_(False)
            self.model.classif.requires_grad_(True)
            self.optimizer_D.zero_grad() # 鉴别器优化器
           
            pred, dis_labels, yl_label, dis_out, mape_data= self.model(X, label, sample_list, yl_values, self._iter_cnt)
            loss2_2 = self.criterion(dis_out.reshape(b,t,n,-1), yl_label.float()) # 鉴别器损失
            
            loss2_2.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.D_params, self._clip_grad_value)
            self.optimizer_D.step()

            train_loss.append(loss1.item()) # 效用loss，区域mae
            train_mape.append(mape) # 区域mape
            train_rmse.append(rmse) # 区域rmse
            dis_loss.append(loss2_2.item()) # 鉴别器的loss
            static_fair.append(loss3.item()) # 区域静态损失
            dynamic_fair.append(loss4.item()) # 个体动态损失

            loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, Loss1: {:.4f}, Dis Loss: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'
            self._logger.info(loss_message.format(epoch + 1, batch_count+1, loss1, loss2_2, loss3, loss4))

            '''鉴别器 间隔采样'''
            # if (batch_count+1) % 10 == 0:

            #     self.model.encoder.requires_grad_(False)
            #     self.model.decoder.requires_grad_(False)
            #     self.model.classif.requires_grad_(True)
            #     self.optimizer_D.zero_grad() # 鉴别器优化器
           
            #     pred, dis_labels, yl_label, dis_out, mape_data= self.model(X, label, sample_list, yl_values, self._iter_cnt)
            #     # loss1 = self._loss_fn(new_pred, new_label, mask_value) # 效用损失，区域的mae
            #     loss2_2 = self.criterion(dis_out.reshape(b,t,n,-1), yl_label.float()) # 鉴别器损失
            #     # loss3, values_region = static_cal(new_pred, new_label) # 静态公平正则化
            #     # loss4 = dynamic_cal5_global(dis_labels, dis_out, sample_map, district_nodes)

            #     # print("44. requires_grad:", pred.requires_grad, dis_labels.requires_grad, yl_label.requires_grad, dis_out.requires_grad)  
            #     # print("77. requires_grad:", loss2_2.requires_grad)
            #     loss2_2.backward()
            #     if self._clip_grad_value != 0:
            #         torch.nn.utils.clip_grad_norm_(self.D_params, self._clip_grad_value)
            #     self.optimizer_D.step()

            #     train_loss.append(loss1.item()) # 效用loss，区域mae
            #     train_mape.append(mape) # 区域mape
            #     train_rmse.append(rmse) # 区域rmse
            #     dis_loss.append(loss2_2.item()) # 鉴别器的loss
            #     static_fair.append(loss3.item()) # 区域静态损失
            #     dynamic_fair.append(loss4.item()) # 个体动态损失

            #     loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, Loss1: {:.4f}, Dis Loss: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'
            #     self._logger.info(loss_message.format(epoch + 1, batch_count+1, loss1, loss2_2, loss3, loss4))

            # else:

            #     train_loss.append(loss1.item()) # 效用loss，区域mae
            #     train_mape.append(mape) # 区域mape
            #     train_rmse.append(rmse) # 区域rmse
            #     # dis_loss.append(loss2_2.item()) # 鉴别器的loss
            #     static_fair.append(loss3.item()) # 区域静态损失
            #     dynamic_fair.append(loss4.item()) # 个体动态损失

            #     loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, Loss1: {:.4f}, Dis Loss: 0.0000, Loss3: {:.4f}, Loss4: {:.4f}'
            #     self._logger.info(loss_message.format(epoch + 1, batch_count+1, loss1, loss3, loss4))

            '''重新采样部分'''
            
            # sample_list = optimize_selection(self.district13_road_index, total_nodes, self.sample_num, node_count_global, self.n_number)
            sample_list, node_count_global = optimize_selection(self.district13_road_index, self.sample_num, node_count_global) # greedy专用，在采样过程更新global
            # sorted_dict = pro_sample2(values_region, values_node, self.district13_road_index)
            # sample_list = resampel(sorted_dict,self.sample_num,self.district13_road_index)
            sample_map = sum_map(sample_list, self.sample_num)
            sample_dict = sample_map_district(sample_list, self.district13_road_index)
            district_nodes = get_district_nodes(sample_dict, sample_map)
            

            self._iter_cnt += 1
            batch_count += 1
        
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse), np.mean(dis_loss), np.mean(static_fair), np.mean(dynamic_fair) # 一个epoch内！多个batch所以mean
    



    def train(self, sample_list, sample_map, sample_dict, district_nodes, yl_values):
        self._logger.info('Start training!') #

        wait = 0
        min_loss = np.inf
        total_nodes = sum(list(self.district13_road_index.values()), []) # 所有节点的list，[0-937]
        for epoch in range(self._max_epochs):
            t1 = time.time()
            mtrain_loss, mtrain_mape, mtrain_rmse, mtraind_loss, mtrain_sfair, mtrain_dfair = self.train_batch(sample_list, sample_map, sample_dict, district_nodes, yl_values, epoch, total_nodes)
            t2 = time.time()

            v1 = time.time()
            self._logger.info("==========validation and test===============")
            mvalid_loss, mvalid_mape, mvalid_rmse, mvalidd_loss, mvalid_sfair, mvalid_dfair = self.evaluate('val',sample_list, sample_map, sample_dict, district_nodes, yl_values, epoch, total_nodes)
            v2 = time.time()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            # 为鉴别器
            if self.scheduler_D is None:
                cur_lr_d = self.d_lrate
            else:
                cur_lr_d = self.scheduler_D.get_last_lr()[0]
                self.scheduler_D.step()

            # 每个epoch，train一回，valid一回
            message = 'Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train dis_loss: {:.4f}, Train SFair: {:.4f}, Train DFair: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Valid dis_loss: {:.4f}, Valid SFair: {:.4f}, Valid DFair: {:.4f}, Train Time: {:.4f}s/epoch, Valid Time: {:.4f}s, LR: {:.4e}, D_LR: {:.4e}'
            self._logger.info(message.format(epoch + 1, mtrain_loss, mtrain_rmse, mtrain_mape, mtraind_loss, mtrain_sfair, mtrain_dfair, \
                                             mvalid_loss, mvalid_rmse, mvalid_mape, mvalidd_loss, mvalid_sfair, mvalid_dfair, \
                                             (t2 - t1), (v2 - v1), cur_lr, cur_lr_d))

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
        
        self.evaluate('test',sample_list, sample_map, sample_dict, district_nodes, yl_values, epoch, total_nodes)



    # 直接用分层采样，看会不会效果更好. evaluate 和 train函数一样！！（模仿）
    def evaluate(self, mode, sample_list, sample_map, sample_dict, district_nodes, yl_values,epoch,total_nodes):
        if mode == 'test': # 如果是测试，加载模型
            self.load_model(self._save_path)
        self.model.eval()

        preds = []
        labels = []
        e_loss = []
        e_mape = []
        e_rmse = []
        edis_loss, estatic_fair, edynamic_fair = [], [],[]
        node_count_global = {}
        node_count_global = calcute_global_dic(node_count_global, sample_list) # greedy专用

        batch_count = 0
        with torch.no_grad():
            for X, label in self._dataloader[mode + '_loader'].get_iterator():
                # print(batch_count, sample_list)
                # if batch_count == 2:
                #     break

                # self._logger.info(sample_list) # 打印新的采样
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
                loss2 = self.criterion(dis_out.reshape(b,t,n,-1), yl_label.float()) # 鉴别器损失，450个节点的准确优劣分类
                loss3, values_region = static_cal(new_pred, new_label) # 静态公平正则化
                # loss4, node_count_global = dynamic_cal5_global_xhbl(dis_labels, dis_out, sample_map, district_nodes, self.district13_road_index, node_count_global) # xhbl更新了node_count_global
                loss4 = dynamic_cal5_global(dis_labels, dis_out, sample_map, district_nodes) # greedy专用
                
                loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, Loss1: {:.4f}, Dis Loss: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'
                self._logger.info(loss_message.format(epoch + 1, batch_count+1, loss1, loss2, loss3, loss4))

                e_loss.append(loss1.item())
                e_mape.append(mape)
                e_rmse.append(rmse)
                edis_loss.append(loss2.item())
                estatic_fair.append(loss3.item())
                edynamic_fair.append(loss4.item())

                # sample_list = optimize_selection(self.district13_road_index, total_nodes, self.sample_num, node_count_global, self.n_number) # xhbl
                sample_list, node_count_global = optimize_selection(self.district13_road_index, self.sample_num, node_count_global) # greedy，在采用中更新了global
                sample_map = sum_map(sample_list, self.sample_num)
                sample_dict = sample_map_district(sample_list, self.district13_road_index)
                district_nodes = get_district_nodes(sample_dict, sample_map)
            


                self._iter_cnt += 1
                batch_count += 1
                

        if mode == 'val':
            # mae = self._loss_fn(preds, labels, mask_value).item()
            # mape = masked_mape(preds, labels, mask_value).item()
            # rmse = masked_rmse(preds, labels, mask_value).item()
            mvalid_loss, mvalid_mape, mvalid_rmse, mvalidd_loss, mvalid_sfair, mvalid_dfair = \
                np.mean(e_loss), np.mean(e_mape), np.mean(e_rmse), np.mean(edis_loss), np.mean(estatic_fair), np.mean(edynamic_fair)
                
            return mvalid_loss, mvalid_mape, mvalid_rmse, mvalidd_loss, mvalid_sfair, mvalid_dfair

        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []
            test_sfair, test_dfair = [],[]

            log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test D_Loss: {:.4f}, Test SFair: {:.4f}, Test DFair: {:.4f}'
            self._logger.info(log.format(np.mean(e_loss), np.mean(e_rmse), np.mean(e_mape), np.mean(edis_loss), np.mean(estatic_fair), np.mean(edynamic_fair)))
