'''
程序运行5k+
带有T的计算动态的loss4, loss和loss2均有retaion
没有T的每个batch计算loss4，loss2没有retain
T1：只有1个list高效，dis_loss上升
T2：有3个list不高效，dis_loss下降
'''

import torch
from src.base.engine import BaseEngine
# from src.utils.metrics import masked_mape, masked_rmse
from src.utils.metrics import masked_mape, masked_rmse, masked_mpe, masked_mae
from src.utils.metrics_region import masked_mae2
from src.engines.sample_T_single import * # 计算loss3和loss4的函数
from src.engines.sample_optimal_greedy_T import *  # 先区域后节点

import time
import numpy as np
# from src.engines.IDF_baselines import * # 未添加公平框架时用
import pickle

# # 希望输出是每个区域一个(b,t,1)
# def static_cal(pred,label): # (b, t, 13, 1), 13表示区域数量，1表示预测的交通值维度
#     b, t = pred.shape[0], pred.shape[1]
#     pred = pred.reshape(b * t, -1) # 将 pred 和 label 转换为形状为 (b*t, 13)
#     label = label.reshape(b * t, -1)

#     # 初始化一个张量用于保存每个区域对之间的 MAPE 差值
#     mape_diff = [] # torch.zeros((78,))

#     # 计算 MAPE 差值
#     idx = 0
#     for i in range(pred.shape[1]-1): # 区域数量-1-->13-1=12 # i = 0123456789--11
#         mape_i = torch.abs(pred[:, i] - label[:, i]) / label[:, i] # 区域i的mape
            
#         for j in range(i + 1, pred.shape[1]):
#             mape_j = torch.abs(pred[:, j] - label[:, j]) / label[:, j] # 区域j的mape
#             # mape_diff[idx] = torch.mean(mape_i - mape_j)
#             # mape_diff.append(torch.abs(torch.sum(mape_i - mape_j))) # mean换成sum 
#             mape_diff.append(torch.sum(torch.abs(mape_i - mape_j)))
#             idx += 1
#         # print("--------------------:",mape_i, mape_j,mape_j.shape)
        
#     mape_diff_mean = torch.mean(torch.stack(mape_diff), dim=0)

#     return mape_diff_mean

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

def give_yl_label2(mape_data, yl_values): # yl_values阈值，大于表示误差大判断为l，小于表示误差小判断为y
    # mape_data[mape_data > yl_values] = -1
    # mape_data[mape_data <= yl_values] = 1 # mape_data是(b,n)
    yl_label = torch.where(mape_data > yl_values, 0, 1) # 误差大就0！表示牺牲
    return yl_label

class ASTGCN_Engine(BaseEngine):
    def __init__(self, criterion, d_lrate, n_number,model_params, D_params, optimizer_D, scheduler_D, a_loss3, a_loss4, T_dynamic, sample_num, district13_road_index, **args):
        super(ASTGCN_Engine, self).__init__(**args)
        for p in self.model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)

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
        filename = 'final_model_s{}_HK3_1.pt'.format(self._seed)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))
    
    def load_model(self, save_path):
        filename = 'final_model_s{}_HK3_1.pt'.format(self._seed)
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename)))

    def train_batch(self, time_T, sample_list, sample_map, sample_dict, district_nodes, yl_values, epoch, total_nodes):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        train_mape2, train_mpe2, train_mae2 = [],[],[]
        
        dis_loss =[]
        benefit_list, equal_list, sarcifice_list =[],[],[]
        b_mean_list,s_mean_list= [],[]

        static_fair, dynamic_fair = [],[]
        batch_count = 0

        node_count_global = {} # 全局，贯穿1个epoch内所有batch记录采样过的个体情况，采样次数
        node_count_global = calcute_global_dic(node_count_global, sample_list)

        yl_node_dic_list = [0,0,0]

        self._dataloader['train_loader'].shuffle()
        for X, label in self._dataloader['train_loader'].get_iterator():
            self._optimizer.zero_grad()
            self.optimizer_D.zero_grad()

            # X (b, t, n, f), label (b, t, n, 1)
            X, label = self._to_device(self._to_tensor([X, label]))
            X, label= X[:, :, sample_list, :], label[:, :, sample_list, :] # (b,t,938,ci)--->(b,t,450,ci)
            b,t,n,c = X.shape # n是采样的数目

            pred, dis_out = self.model(X, label)
            pred, label = self._inverse_transform([pred, label])
            # 不用反归一化，因为在data/generate_data_for_training.py中归一化后所有数据为0，注释了归一化的代码
            
            new_pred = torch.zeros(b, t, len(list(district_nodes.keys())), c)
            new_label = torch.zeros(b, t, len(list(district_nodes.keys())), c)
            for region, nodes_list in district_nodes.items(): # region是str '0'
                new_pred[:, :, int(region)] = pred[:, :, nodes_list].mean(dim=2)#, keepdim=True)#.mean(dim=2)
                new_label[:, :, int(region)] = label[:, :, nodes_list].mean(dim=2)#, keepdim=True) #.mean(dim=2)
            
            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if new_label.min() < 1:
                mask_value = new_label.min()
            if self._iter_cnt == 0:
                print('check mask value', mask_value)

            node_mask_value = torch.tensor(0)
            if label.min() < 1:
                node_mask_value = label.min()
            if self._iter_cnt == 0:
                print('check single mask value', node_mask_value)


            # 节点级别的mape
            mape2 = masked_mape(pred, label, node_mask_value).item()
            mape_data = calcu_mape2(pred, label, node_mask_value)
            yl_label = give_yl_label2(mape_data, yl_values)
            loss2 = self.criterion(dis_out, yl_label.float()) # 鉴别器损失
            dis_labels = torch.where(dis_out > 0.5, torch.ones_like(dis_out), torch.full_like(dis_out, -1)).reshape(b,t, n, -1) # 使得dis_label具有梯度 .requires_grad_()
            
            loss1 = self._loss_fn(new_pred, new_label, mask_value)
            mape = masked_mape(new_pred, new_label, mask_value).item()
            rmse = masked_rmse(new_pred, new_label, mask_value).item()
            loss3, values_region = static_cal(new_pred, new_label, self._device) # 静态公平正则化
            
            yl_node_dic = get_yl_batch(dis_out, sample_map) # 键0-938，值对应yl
            yl_node_dic_list[batch_count % time_T] = yl_node_dic
            
            '''如果满足动态时间T'''
            if batch_count >= (time_T-1):
                
                # loss4 = dynamic_cal5_global_no5_T(time_T, T_district_nodes, T_sample_map, T_disout)
                loss4, count_nodes = dynamic_cal5_global_T22(time_T, yl_node_dic_list)
                
                loss = loss1 + self.a_loss3*loss3 + self.a_loss4*loss4  # HK 0.1, dis_lobel 0.01
                dynamic_fair.append(loss4.item()) # 个体动态损失
            else:
                loss4 = -1
                loss = loss1 + self.a_loss3*loss3

            loss.backward(retain_graph=True)#retain_graph=True)
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model_params, self._clip_grad_value)
            self._optimizer.step()
            
            loss2.backward(retain_graph=True)
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.D_params, self._clip_grad_value)
            self.optimizer_D.step()
            
            train_loss.append(loss1.item()) # 效用loss，区域mae
            train_mape.append(mape) # 区域mape
            train_mape2.append(mape2) # 区域mape
            train_rmse.append(rmse) # 区域rmse
            dis_loss.append(loss2.item()) # 鉴别器的loss
            static_fair.append(loss3.item()) # 区域静态损失

            loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, mape2:{:.4f}, Loss1: {:.4f}, Dis Loss: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'  # , Loss44: {:.4f}
            self._logger.info(loss_message.format(epoch + 1, batch_count+1, mape2, loss1, loss2, loss3, loss4)) # , loss44))
        
            '''
            重新采样部分：1.每个batch根据12原则，2. 每个batch根据123，3.每个batch根据123，每T个batch根据1234
            '''
            sample_list, node_count_global = optimize_selection(self.district13_road_index, self.sample_num, node_count_global)
            sample_map = sum_map(sample_list, self.sample_num)
            sample_dict = sample_map_district(sample_list, self.district13_road_index)
            district_nodes = get_district_nodes(sample_dict, sample_map)
            
            self._iter_cnt += 1
            batch_count += 1

        return np.mean(train_loss), np.mean(train_mape), np.mean(train_mape2), np.mean(train_rmse), np.mean(dis_loss), np.mean(static_fair), np.mean(dynamic_fair)   # 一个epoch内！多个batch所以mean



    def train(self, time_T, sample_list, sample_map, sample_dict, district_nodes, yl_values):
        self._logger.info('Start training!')

        '''没加公平框架时的每一个epoch的mape'''
        filename = 'ylvalue_s{}_HK_3_node_mape.pkl'.format(self._seed)
        # 从文件中读取两个列表
        with open(os.path.join(self._save_path, filename), 'rb') as f:
            loaded_lists = pickle.load(f)
            train_yllist, val_yllist = loaded_lists

        wait = 0
        min_loss = np.inf
        total_nodes = sum(list(self.district13_road_index.values()), []) # 所有节点的list，[0-937]
        for epoch in range(self._max_epochs):

            train_index, val_index = min(len(train_yllist)-1, epoch), min(len(val_yllist)-1, epoch)
            yl_values = train_yllist[train_index]
            self._logger.info(yl_values)
            print("train: ", yl_values)

            t1 = time.time()
            mtrain_loss, mtrain_mape, mtrain_mape2, mtrain_rmse, mtraind_loss, mtrain_sfair, mtrain_dfair\
                  = self.train_batch(time_T, sample_list, sample_map, sample_dict, district_nodes, yl_values, epoch, total_nodes)
            t2 = time.time()

            yl_values = val_yllist[val_index]
            self._logger.info(yl_values)
            print("valid: ", yl_values)

            v1 = time.time()
            self._logger.info("==========validation and test===============")
            mvalid_loss, mvalid_mape, mvalid_mape2, mvalid_rmse, mvalidd_loss, mvalid_sfair, mvalid_dfair\
                  = self.evaluate('val',time_T, sample_list, sample_map, sample_dict, district_nodes, yl_values, epoch, total_nodes)
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
            message = 'Epoch: {:03d}, Train, mape2: {:.4f}, Loss: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}, dis_loss: {:.4f}, SFair: {:.4f}, DFair: {:.4f}, Time: {:.4f}s/epoch, LR: {:.4e}, D_LR: {:.4e}'
            self._logger.info(message.format(epoch + 1, mtrain_mape2, mtrain_loss, mtrain_rmse, mtrain_mape, mtraind_loss, mtrain_sfair, mtrain_dfair, \
                                             (t2 - t1), cur_lr, cur_lr_d))
            message = 'Epoch: {:03d}, Test, mape2: {:.4f},  Loss: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}, dis_loss: {:.4f}, SFair: {:.4f}, DFair: {:.4f}, Time: {:.4f}s, LR: {:.4e}, D_LR: {:.4e}'
            self._logger.info(message.format(epoch + 1, mvalid_mape2, mvalid_loss, mvalid_rmse, mvalid_mape, mvalidd_loss, mvalid_sfair, mvalid_dfair,\
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
        
        print("test:", yl_values)
        self._logger.info(yl_values)
        self.evaluate('test',time_T, sample_list, sample_map, sample_dict, district_nodes, yl_values, epoch, total_nodes)



    def evaluate(self, mode, time_T, sample_list, sample_map, sample_dict, district_nodes, yl_values,epoch,total_nodes):
        if mode == 'test': # 如果是测试，加载模型
            self.load_model(self._save_path)
        self.model.eval()

        loss_region_list = []

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
        node_count_global = calcute_global_dic(node_count_global, sample_list)

        yl_node_dic_list = [0,0,0]

        batch_count = 0
        with torch.no_grad():
            for X, label in self._dataloader[mode + '_loader'].get_iterator():
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                X, label= X[:, :, sample_list, :], label[:, :, sample_list, :] # (b,t,938,ci)--->(b,t,450,ci)
                b,t,n,c = X.shape # n是采样的数目,450

                pred, dis_out = self.model(X, label)
                pred, label = self._inverse_transform([pred, label])
                
                new_pred = torch.zeros(b, t, len(list(district_nodes.keys())), c)
                new_label = torch.zeros(b, t, len(list(district_nodes.keys())), c)
                for region, nodes_list in district_nodes.items(): # region是str '0' # sample2.py中所有round+0.5确保0-12均有采样，即有0-12
                    new_pred[:, :, int(region)] = pred[:, :, nodes_list].mean(dim=2)#, keepdim=True)#.mean(dim=2)
                    new_label[:, :, int(region)] = label[:, :, nodes_list].mean(dim=2)#, keepdim=True) #.mean(dim=2)

                # 针对13个区域
                mask_value = torch.tensor(0)
                if new_label.min() < 1:
                    mask_value = new_label.min()
                if self._iter_cnt == 0:
                    print('check mask value', mask_value)

                # 针对450个采样节点
                mask_value_single = torch.tensor(0)
                if label.min() < 1:
                    mask_value_single = label.min()
                if self._iter_cnt == 0:
                    print('check single mask value', mask_value_single)


                '''看13个区域的情况'''
                loss_region = masked_mae2(new_pred, new_label, mask_value)

                mape2 = masked_mape(pred, label, mask_value_single).item()
                mape_data = calcu_mape2(pred, label, mask_value_single)
                yl_label = give_yl_label2(mape_data, yl_values)
                dis_labels = torch.where(dis_out > 0.5, torch.ones_like(dis_out), torch.full_like(dis_out, -1)).reshape(b,t, n, -1) # 使得dis_label具有梯度 .requires_grad_()
                
                loss2 = self.criterion(dis_out, yl_label.float()) # 鉴别器损失

                loss1 = self._loss_fn(new_pred, new_label, mask_value)
                mape = masked_mape(new_pred, new_label, mask_value).item()
                rmse = masked_rmse(new_pred, new_label, mask_value).item()
                loss3, values_region = static_cal(new_pred, new_label, self._device) # 静态公平正则化

                yl_node_dic = get_yl_batch(dis_out, sample_map) # 键0-938，值对应yl
                yl_node_dic_list[batch_count % time_T] = yl_node_dic
                
            
                '''如果满足动态时间T'''
                if batch_count >= (time_T-1):
                
                    # loss4 = dynamic_cal5_global_no5_T(time_T, T_district_nodes, T_sample_map, T_disout)
                    loss4, count_nodes = dynamic_cal5_global_T22(time_T, yl_node_dic_list)
                
                    edynamic_fair.append(loss4.item()) # 个体动态损失
                else:
                    loss4 = -1

                e_loss.append(loss1.item())
                e_mape.append(mape)
                e_mape2.append(mape2)
                e_rmse.append(rmse)
                edis_loss.append(loss2.item())
                estatic_fair.append(loss3.item())

                if mode == "test":
                    loss_region_list.append(loss_region)

                loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, mape2:{:.4f}, Loss1: {:.4f}, Dis Loss: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'  # , Loss44: {:.4f}
                self._logger.info(loss_message.format(epoch + 1, batch_count+1, mape2, loss1, loss2, loss3, loss4)) # , loss44))

                sample_list, node_count_global = optimize_selection(self.district13_road_index, self.sample_num, node_count_global)
                # sample_list = optimize_selection(self.district13_road_index, total_nodes, self.sample_num, node_count_global, self.n_number)
                sample_map = sum_map(sample_list, self.sample_num)
                sample_dict = sample_map_district(sample_list, self.district13_road_index)
                district_nodes = get_district_nodes(sample_dict, sample_map)

                self._iter_cnt += 1
                batch_count += 1

        if mode == 'val':
            mvalid_loss, mvalid_mape, mvalid_mape2, mvalid_rmse, mvalidd_loss, mvalid_sfair, mvalid_dfair\
                  = np.mean(e_loss), np.mean(e_mape), np.mean(e_mape2), np.mean(e_rmse), np.mean(edis_loss), np.mean(estatic_fair), np.mean(edynamic_fair)

            return mvalid_loss, mvalid_mape, mvalid_mape2, mvalid_rmse, mvalidd_loss, mvalid_sfair, mvalid_dfair

        elif mode == 'test':
    
            # log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test SFair: {:.4f}, Test DFair: {:.4f}'
            # self._logger.info(log.format(np.mean(e_loss), np.mean(e_rmse), np.mean(e_mape), np.mean(estatic_fair), np.mean(edynamic_fair)))
            log = 'Average Test mape2: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}, D_Loss: {:.4f}, SFair: {:.4f}, DFair: {:.4f}.'
            self._logger.info(log.format(np.mean(e_mape2), np.mean(e_loss), np.mean(e_rmse), np.mean(e_mape), np.mean(edis_loss), np.mean(estatic_fair), np.mean(edynamic_fair)))
            loss_region_tensor = torch.stack(loss_region_list,dim=0)
            print(loss_region_tensor.shape)
            mean_values = loss_region_tensor.mean(dim=0)
            print(mean_values)
            self._logger.info(mean_values)