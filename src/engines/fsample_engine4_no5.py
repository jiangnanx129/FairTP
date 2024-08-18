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
# from src.engines.sample_optimal_greedy_test import * # 循环遍历实现优化采样
from src.engines.sample_optimal_greedy_test2 import * 
import pickle
'''
10月份新讨论，未加动静态约束于采样，将采样视为优化问题
1. 缓解数据不平衡  2. 确保采样的多样性
关于fsample4，dis_out是(n,2)
'''

def calcu_mape2(preds, labels, null_val): # 均为(b,t,n,c)-gpu, 在region_map_test.py中有测试
    # loss = torch.abs((preds - labels) / labels) # (b,t,n,c)
    # #loss = torch.mean(loss, dim=(3)) # (b,t,n)

    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return loss 

# 0/1牺牲
def give_yl_label2(mape_data, yl_values): # yl_values阈值，大于表示误差大判断为l，小于表示误差小判断为y
    # mape_data[mape_data > yl_values] = -1
    # mape_data[mape_data <= yl_values] = 1 # mape_data是(b,n)
    yl_label = torch.where(mape_data > yl_values, 0, 1) # 误差大就0！表示牺牲
    return yl_label


class FSAMPE_Engine(BaseEngine):
    def __init__(self, criterion, d_lrate, n_number,model_params, D_params, optimizer_D, scheduler_D, a_loss3, a_loss4,  T_dynamic, sample_num, district13_road_index, **args):
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
        filename = 'final_model_s{}_HK3_disout7_8i7.pt'.format(self._seed)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))
    
    def load_model(self, save_path):
        filename = 'final_model_s{}_HK3_disout7_8i7.pt'.format(self._seed)
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename)))
        
    # 单个epoch
    def train_batch(self, sample_list, sample_map, sample_dict, district_nodes, yl_values, epoch, total_nodes):
        self.model.train()

        train_loss = []
        train_mape = []
        train_mape2 = []
        train_rmse = []
        dis_loss, static_fair, dynamic_fair =[],[],[]
        
        benefit_list, equal_list, sarcifice_list =[],[],[]
        b_mean_list,s_mean_list= [],[]
        self._dataloader['train_loader'].shuffle()
        batch_count = 0
        node_count_global = {} # 全局，贯穿1个epoch内所有batch记录采样过的个体情况，采样次数
        node_count_global = calcute_global_dic(node_count_global, sample_list) # greedy专用

        # dis_out_list, sample_dict_list, sample_map_list, district_nodes_list, sample_list_list = [0 for _ in range(self.T_dynamic)],\
        #     [0 for _ in range(self.T_dynamic)],[0 for _ in range(self.T_dynamic)],[0 for _ in range(self.T_dynamic)], [0 for _ in range(self.T_dynamic)] # 先装入初始化的数据
        for X, label in self._dataloader['train_loader'].get_iterator(): # 循环每一个batch
            
            '''
            co-train的两个loss
            参考：https://blog.csdn.net/qq_40737596/article/details/127674436 和 https://zhuanlan.zhihu.com/p/596917587
            1. loss.backward(retain_graph=True)，不释放中间变量的梯度，是鉴别器的输入
            2. 除了1，再加上在fsample3.py文件中，给中间变量加上.detach()： dcgru4_state_tensor.reshape(t,b,n,-1).detach()
            '''
            
            # self._logger.info(sample_list) # 打印新的采样
            self._optimizer.zero_grad() # encoder和decoder
            self.optimizer_D.zero_grad() # 鉴别器优化器

            X, label = self._to_device(self._to_tensor([X, label]))
            X, label= X[:, :, sample_list, :], label[:, :, sample_list, :] # 已划分好batch，先采样，再进model。(b,t,938,ci)--->(b,t,450,ci)
            
            pred, dis_out = self.model(X, label, sample_list, yl_values, self._iter_cnt)
            # print("55. requires_grad:", pred.requires_grad, dis_labels.requires_grad, yl_label.requires_grad, dis_out.requires_grad) # TFFT
            '''
            pred为反归一化的预测结果，(b,t,n,cout)
            dis_labels为鉴别器输出结果, (b,t,n,cout), 全1或-1
            yl_label为优劣label，根据o和ground truth的比较得出, (b,t,n,cout), 全0或1表示牺牲/受益
            dis_out为鉴别器输出，(t*b*n,cout), 经过sigmoid为0-1的值
            mape_data是根据o和ground truth的比较得出的mape数据值，全为正数，(b,t,n,cout)
            '''
            pred, label = self._inverse_transform([pred, label]) # pred取值40+
            
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
            
            '''
            mape2为节点级的mape，比区域级的mape要大
            '''
            node_mask_value = torch.tensor(0)
            if label.min() < 1:
                node_mask_value = label.min()
            if self._iter_cnt == 0:
                print('check mask value', node_mask_value)

            '''
            根据模型输出，计算(b,t,n,c)维度的mape_data, 依据yl_label给出新的label，以方便鉴别器操作
            '''
            mape2 = masked_mape(pred, label, node_mask_value).item() # 这就是yl_label
            mape_data = calcu_mape2(pred, label,node_mask_value)
            yl_label = give_yl_label2(mape_data, mape2)
            dis_labels = torch.where(dis_out > 0.5, torch.ones_like(dis_out), torch.full_like(dis_out, -1)).reshape(b,t, n, -1) # 使得dis_label具有梯度 .requires_grad_()
        

            loss2 = self.criterion(dis_out, yl_label.float()) # 鉴别器损失，450个节点的准确优劣分类
            
            loss3, values_region = static_cal(new_pred, new_label, self._device) # 静态公平正则化
            loss4, greater_than_zero, less_than_zero, equal_to_zero\
                = dynamic_cal5_global_no5_2(dis_labels, dis_out, sample_map, district_nodes) # , self.district13_road_index, node_count_global) # 更新了node_count_global
            # loss4, node_count_global = dynamic_cal5_global_xhbl(dis_labels, dis_out, sample_map, district_nodes, self.district13_road_index, node_count_global) # 更新了node_count_global
            # print("loss5:", loss5, loss5.requires_grad)
           
            '''loss4基于dis_out,但是是-1/1'''
            # loss4, node_count_global = dynamic_cal5_global2(dis_labels, dis_out, sample_map, district_nodes, self.district13_road_index, node_count_global) # 更新了node_count_global
            # loss4, loss44, node_count_global = dynamic_cal5_global3(dis_labels, dis_out, sample_map, district_nodes, self.district13_road_index, node_count_global) # 更新了node_count_global
            # print("66. requires_grad:", loss1.requires_grad, loss2.requires_grad, loss3.requires_grad, loss4.requires_grad) # TTTT
          
            # loss = loss1 + 5*loss2 + 0.05*loss3 + 0.1*loss4 # test 40, 0.1,0.15 3 SD
            # print("11. loss2.requires_grad:", loss2.requires_grad)
            
            benefit_list.append(greater_than_zero) # list中每个元素表示，该次batch牺牲的节点数目
            equal_list.append(equal_to_zero) 
            sarcifice_list.append(less_than_zero)  


            '''
            2部分损失
            1. loss+loss3+loss4：效用+区域静态+个体动态
            2. loss2：鉴别器损失
            '''
            loss = loss1 + self.a_loss3*loss3 + self.a_loss4*loss4 # + loss2  # HK 0.1, dis_lobel 0.01
            # loss = loss1 + 0.05*loss2 + 0.01*loss3 + 0.02*loss4 # HK2， o做输入。
            
            # print("loss2.requires_grad:", loss.requires_grad)
            # print("loss4.requires_grad:", loss.requires_grad)
            

            loss.backward(retain_graph=True) # retain_graph=True) 不释放中间变量的梯度，是鉴别器的输入
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model_params, self._clip_grad_value)
            self._optimizer.step()

            loss2.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.D_params, self._clip_grad_value)
            self.optimizer_D.step()

            train_loss.append(loss1.item()) # 效用loss，区域mae
            train_mape.append(mape) # 区域mape
            train_mape2.append(mape2) # 区域mape
            train_rmse.append(rmse) # 区域rmse
            dis_loss.append(loss2.item())
            static_fair.append(loss3.item()) # 区域静态损失
            # dynamic_fair.append(loss4.item()) # 个体动态损失
            dynamic_fair.append(loss4.item())
        

            # loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, Loss1: {:.4f}, Dis Loss: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'
            # self._logger.info(loss_message.format(epoch + 1, batch_count+1, loss1, loss2, loss3, loss4))
            loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, mape2:{:.4f}, Loss1: {:.4f}, Dis Loss: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f},  Benefit: {:.4f}, Sacrifice: {:.4f}, Equal: {:.4f}'  # , Loss44: {:.4f}
            self._logger.info(loss_message.format(epoch + 1, batch_count+1, mape2, loss1, loss2, loss3, loss4,  greater_than_zero, less_than_zero, equal_to_zero)) # , loss44))

            # loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, Bmax: {:.4f}, Bmin: {:.4f},Bmea: {:.4f}, Ber: {:.4f}, Smax: {:.4f}, Smin: {:.4f}, Smea: {:.4f},Ser: {:.4f}'  # , Loss44: {:.4f}
            # self._logger.info(loss_message.format(epoch + 1, batch_count+1, gmax_value, gmin_value, gav,gerror,  lmax_value, lmin_value,lav, lerror)) # , loss44))

            '''先不看具体的值，看平均值'''

             


            '''重新采样部分'''
            sample_list, node_count_global = optimize_selection(self.district13_road_index, self.sample_num, node_count_global)
                
            # sample_list = optimize_selection(self.district13_road_index, total_nodes, self.sample_num, node_count_global, self.n_number) # xhbl的采样方式
            # sorted_dict = pro_sample2(values_region, values_node, self.district13_road_index)
            # sample_list = resampel(sorted_dict,self.sample_num,self.district13_road_index)
            sample_map = sum_map(sample_list, self.sample_num)
            sample_dict = sample_map_district(sample_list, self.district13_road_index)
            district_nodes = get_district_nodes(sample_dict, sample_map)
            

            self._iter_cnt += 1
            batch_count += 1
        
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_mape2), np.mean(train_rmse), np.mean(dis_loss), np.mean(static_fair), np.mean(dynamic_fair) \
            , np.mean(benefit_list), np.mean(sarcifice_list), np.mean(equal_list)   # 一个epoch内！多个batch所以mean




    def train(self, sample_list, sample_map, sample_dict, district_nodes, yl_values):
        self._logger.info('Start training!') #
        


        wait = 0
        min_loss = np.inf
        total_nodes = sum(list(self.district13_road_index.values()), []) # 所有节点的list，[0-937]
        for epoch in range(self._max_epochs):
            

            t1 = time.time()
            mtrain_loss, mtrain_mape, mtrain_mape2, mtrain_rmse, mtraind_loss, mtrain_sfair, mtrain_dfair, mtrain_be, mtrain_sa, mtrain_eq\
                  = self.train_batch(sample_list, sample_map, sample_dict, district_nodes, yl_values, epoch, total_nodes)
            t2 = time.time()


            v1 = time.time()
            self._logger.info("==========validation and test===============")
            mvalid_loss, mvalid_mape, mvalid_mape2, mvalid_rmse, mvalidd_loss, mvalid_sfair, mvalid_dfair, mvalid_be, mvalid_sa, mvalid_eq\
                  = self.evaluate('val',sample_list, sample_map, sample_dict, district_nodes, yl_values, epoch, total_nodes)
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
            message = 'Epoch: {:03d}, Train, mape2: {:.4f}, Loss: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}, dis_loss: {:.4f}, SFair: {:.4f}, DFair: {:.4f}, Be {:.4f}, Sa {:.4f}, Eq {:.4f}, Time: {:.4f}s/epoch, LR: {:.4e}, D_LR: {:.4e}'
            self._logger.info(message.format(epoch + 1, mtrain_mape2, mtrain_loss, mtrain_rmse, mtrain_mape, mtraind_loss, mtrain_sfair, mtrain_dfair, \
                                             mtrain_be, mtrain_sa, mtrain_eq,\
                                             (t2 - t1), cur_lr, cur_lr_d))
            message = 'Epoch: {:03d}, Test, mape2: {:.4f},  Loss: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}, dis_loss: {:.4f}, SFair: {:.4f}, DFair: {:.4f}, Be {:.4f}, Sa {:.4f}, Eq {:.4f}, Time: {:.4f}s, LR: {:.4e}, D_LR: {:.4e}'
            self._logger.info(message.format(epoch + 1, mvalid_mape2, mvalid_loss, mvalid_rmse, mvalid_mape, mvalidd_loss, mvalid_sfair, mvalid_dfair,\
                                             mvalid_be, mvalid_sa, mvalid_eq,\
                                             (v2 - v1), cur_lr, cur_lr_d))

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
        e_mape2 = []
        e_rmse = []
        edis_loss, estatic_fair, edynamic_fair = [], [],[]
        
        ebenefit_list, eequal_list, esarcifice_list =[],[],[]
        eb_mean_list, es_mean_list= [],[]
        
        node_count_global = {}
        node_count_global = calcute_global_dic(node_count_global, sample_list) # greedy专用

        batch_count = 0
        with torch.no_grad():
            for X, label in self._dataloader[mode + '_loader'].get_iterator():

                # self._logger.info(sample_list) # 打印新的采样
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                X, label= X[:, :, sample_list, :], label[:, :, sample_list, :] # (b,t,938,ci)--->(b,t,450,ci)
        
                pred, dis_out = self.model(X, label, sample_list, yl_values, self._iter_cnt)
                pred, label = self._inverse_transform([pred, label])


                b,t,n,c = X.shape
                new_pred = torch.zeros(b, t, len(list(district_nodes.keys())), c)
                new_label = torch.zeros(b, t, len(list(district_nodes.keys())), c)
                for region, nodes_list in district_nodes.items(): # region是str '0' # sample2.py中所有round+0.5确保0-12均有采样，即有0-12
                    new_pred[:, :, int(region)] = pred[:, :, nodes_list].mean(dim=2)#, keepdim=True)#.mean(dim=2)
                    new_label[:, :, int(region)] = label[:, :, nodes_list].mean(dim=2)#, keepdim=True) #.mean(dim=2)

                mask_value = torch.tensor(0)
                if new_label.min() < 1:
                    mask_value = new_label.min()
                single_mask_value = torch.tensor(0)
                if label.min() < 1:
                    single_mask_value = label.min()

                '''
                根据模型输出，计算(b,t,n,c)维度的mape_data, 依据yl_label给出新的label，以方便鉴别器操作
                '''
                mape2 = masked_mape(pred, label, single_mask_value).item() # 这就是yl_label, 一个值
                mape_data = calcu_mape2(pred, label, single_mask_value) # 返回(b,t,n,c)
                yl_label = give_yl_label2(mape_data, mape2)
                dis_labels = torch.where(dis_out > 0.5, torch.ones_like(dis_out), torch.full_like(dis_out, -1)).reshape(b,t, n, -1) # 使得dis_label具有梯度 .requires_grad_()
        

                loss1 = self._loss_fn(new_pred, new_label, mask_value) # 效用损失
                mape = masked_mape(new_pred, new_label, mask_value).item()
                rmse = masked_rmse(new_pred, new_label, mask_value).item()
                # dis_out.reshape(-1,2), yl_label.reshape(-1,).long() # softmax
                loss2 = self.criterion(dis_out, yl_label.float()) # 鉴别器损失，450个节点的准确优劣分类
                # loss2 = self.criterion(torch.sum(dis_out.reshape(b,t,n,-1), dim=(0, 1, 3)), torch.sum(yl_label.float(), dim=(0, 1, 3)))
            
                loss3, values_region = static_cal(new_pred, new_label, self._device) # 静态公平正则化
                # loss4, greater_than_zero, less_than_zero, equal_to_zero = dynamic_cal5_global(dis_labels, dis_out, sample_map, district_nodes) # , self.district13_road_index, node_count_global) # 更新了node_count_global
                loss4, greater_than_zero, less_than_zero, equal_to_zero\
                        = dynamic_cal5_global_no5_2(dis_labels, dis_out, sample_map, district_nodes) # , self.district13_road_index, node_count_global) # 更新了node_count_global
            
                # loss4, node_count_global = dynamic_cal5_global_xhbl(dis_labels, dis_out, sample_map, district_nodes, self.district13_road_index, node_count_global) # 更新了node_count_global
            
                # loss4, loss44, node_count_global = dynamic_cal5_global3(dis_labels, dis_out, sample_map, district_nodes, self.district13_road_index, node_count_global) # 更新了node_count_global
                
                # loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, Loss1: {:.4f}, Dis Loss: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'
                # self._logger.info(loss_message.format(epoch + 1, batch_count+1, loss1, loss2, loss3, loss4))
                ebenefit_list.append(greater_than_zero) # list中每个元素表示，该次batch牺牲的节点数目
                eequal_list.append(equal_to_zero) 
                esarcifice_list.append(less_than_zero) 


                # loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, Loss1: {:.4f}, Dis Loss: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}, Benefit: {:.4f}, Sacrifice: {:.4f}, Equal: {:.4f}, Be: {:.4f}, Sa: {:.4f}, Eq: {:.4f}'  # , Loss44: {:.4f}
                # self._logger.info(loss_message.format(epoch + 1, batch_count+1, loss1, loss2, loss3, loss4,greater_than_zero, less_than_zero, equal_to_zero, ebenum, esanum,eeqnum)) # , loss44))

                loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, mape2:{:.4f}, Loss1: {:.4f}, Dis Loss: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}, Benefit: {:.4f}, Sacrifice: {:.4f}, Equal: {:.4f}'  # , Loss44: {:.4f}
                self._logger.info(loss_message.format(epoch + 1, batch_count+1, mape2, loss1, loss2, loss3, loss4, greater_than_zero, less_than_zero, equal_to_zero)) # , loss44))

                # loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, Bmax: {:.4f}, Bmin: {:.4f},Bmea: {:.4f}, Ber: {:.4f}, Smax: {:.4f}, Smin: {:.4f}, Smea: {:.4f},Ser: {:.4f}'  # , Loss44: {:.4f}
                # self._logger.info(loss_message.format(epoch + 1, batch_count+1, gmax_value, gmin_value, gav,gerror,  lmax_value, lmin_value,lav, lerror)) # , loss44))

                '''先不看具体的值，看平均值'''

                e_loss.append(loss1.item())
                e_mape.append(mape)
                e_mape2.append(mape2)
                e_rmse.append(rmse)
                edis_loss.append(loss2.item())
                estatic_fair.append(loss3.item())
                # edynamic_fair.append(loss4.item())
                edynamic_fair.append(loss4.item())
                

                sample_list, node_count_global = optimize_selection(self.district13_road_index, self.sample_num, node_count_global)
            
                # sample_list = optimize_selection(self.district13_road_index, total_nodes, self.sample_num, node_count_global, self.n_number)
                sample_map = sum_map(sample_list, self.sample_num)
                sample_dict = sample_map_district(sample_list, self.district13_road_index)
                district_nodes = get_district_nodes(sample_dict, sample_map)
            


                self._iter_cnt += 1
                batch_count += 1
                

        if mode == 'val':
            # mae = self._loss_fn(preds, labels, mask_value).item()
            # mape = masked_mape(preds, labels, mask_value).item()
            # rmse = masked_rmse(preds, labels, mask_value).item()
            mvalid_loss, mvalid_mape, mvalid_mape2, mvalid_rmse, mvalidd_loss, mvalid_sfair, mvalid_dfair, mvalid_be, mvalid_sa, mvalid_eq\
                  = np.mean(e_loss), np.mean(e_mape), np.mean(e_mape2), np.mean(e_rmse), np.mean(edis_loss), np.mean(estatic_fair), np.mean(edynamic_fair)\
                , np.mean(ebenefit_list), np.mean(esarcifice_list), np.mean(eequal_list)   # 一个epoch内！多个batch所以mean

            return mvalid_loss, mvalid_mape, mvalid_mape2, mvalid_rmse, mvalidd_loss, mvalid_sfair, mvalid_dfair , mvalid_be, mvalid_sa, mvalid_eq

            
        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []
            test_sfair, test_dfair = [],[]

            log = 'Average Test mape2: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}, D_Loss: {:.4f}, SFair: {:.4f}, DFair: {:.4f},  Be {:.4f}, Sa {:.4f}, Eq {:.4f}.'
            self._logger.info(log.format(np.mean(e_mape2), np.mean(e_loss), np.mean(e_rmse), np.mean(e_mape), np.mean(edis_loss), np.mean(estatic_fair), np.mean(edynamic_fair)\
                                         , np.mean(ebenefit_list), np.mean(esarcifice_list), np.mean(eequal_list)  ))