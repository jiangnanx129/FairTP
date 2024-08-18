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

class FSAMPE_Engine(BaseEngine):
    def __init__(self, criterion, T_dynamic, sample_num, district13_road_index, **args):
        super(FSAMPE_Engine, self).__init__(**args)
        self.criterion = criterion
        self.T_dynamic = T_dynamic
        self.sample_num = sample_num
        self.district13_road_index = district13_road_index

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = 'final_model_s{}_SDgx.pt'.format(self._seed)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))

    def train_batch(self, sample_list, sample_map, sample_dict, district_nodes, yl_values, epoch):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        static_fair, dynamic_fair = [],[]
        self._dataloader['train_loader'].shuffle()
        batch_count = 0
        # dis_out_list, sample_dict_list, sample_map_list, district_nodes_list, sample_list_list = [0 for _ in range(self.T_dynamic)],\
        #     [0 for _ in range(self.T_dynamic)],[0 for _ in range(self.T_dynamic)],[0 for _ in range(self.T_dynamic)], [0 for _ in range(self.T_dynamic)] # 先装入初始化的数据
        for X, label in self._dataloader['train_loader'].get_iterator(): # 循环每一个batch
            self._optimizer.zero_grad()

            X, label = self._to_device(self._to_tensor([X, label]))
           
            # 已划分好batch，先采样，再进model
            X, label= X[:, :, sample_list, :], label[:, :, sample_list, :] # (b,t,938,ci)--->(b,t,450,ci)
        
            pred, dis_labels, yl_label, dis_out, mape_data = self.model(X, label, sample_list, yl_values, self._iter_cnt)
            label = self._inverse_transform(label) # pred取值40+
            # print(pred.shape, dis_labels.shape, yl_label.shape, dis_out.shape, mape_data.shape)
            # torch.Size([64, 12, 350, 1]) torch.Size([64, 12, 350, 1]) torch.Size([64, 12, 350, 1]) torch.Size([268800, 1]) torch.Size([64, 12, 350, 1])

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
            loss3, values_region = static_cal(new_pred, new_label) # 静态公平正则化
            loss4, values_node = dynamic_cal5(dis_labels, dis_out, sample_map, district_nodes, self.district13_road_index)
            # print(values_region, values_node)
            
            loss = loss1 + 5*loss2 + 0.05*loss3 + 0.1*loss4 # test 40, 0.1,0.15 3 SD
            # loss = loss1 + 2*loss2 + 0.02*loss3 + 0.01*loss4  # HK
            loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'
            self._logger.info(loss_message.format(epoch + 1, batch_count+1, loss1, loss2, loss3, loss4))

            loss.backward() # retain_graph=True)
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss1.item())
            train_mape.append(mape)
            train_rmse.append(rmse)
            static_fair.append(loss3.item())
            dynamic_fair.append(loss4.item())

            sorted_dict = pro_sample2(values_region, values_node, self.district13_road_index)
            sample_list = resampel(sorted_dict,self.sample_num,self.district13_road_index)
            sample_map = sum_map(sample_list, self.sample_num)
            sample_dict = sample_map_district(sample_list, self.district13_road_index)
            district_nodes = get_district_nodes(sample_dict, sample_map)
            

            self._iter_cnt += 1
            batch_count += 1
        
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse), np.mean(static_fair), np.mean(dynamic_fair) # 一个epoch内！多个batch所以mean
    
            # print(loss4,loss3) # 5.48, 602
            # print("区域的概率列表：", region_mape_list) # 700-2000
            # print("节点的概率列表：", softmax_dvalues)

            # node_yl_dic, node_yl_dic_disout, differences_sum_dlabel, differences_sum_dout, softmax_lvalues, softmax_dvalues\
            #       = dynamic_cal5(dis_labels, dis_out, sample_map, district_nodes, self.district13_road_index)
            # print(region_mape_list)
            # print("=================================")
            # aaa = list(node_yl_dic.values())
            # print("最大值和最小值：",max(aaa),min(aaa),aaa[333],aaa[218],aaa[29]) # 506,-42, 230,0,0
            # print("最大值和最小值索引：", aaa.index(max(aaa)), aaa.index(min(aaa))) # 449,639
            # # print(sorted(aaa, reverse=True)) # 从大到小排序
            # has_zero = any(num == 0 for num in aaa)
            # print(has_zero, aaa.count(0))
            # print("=================================")
            # bbb = list(node_yl_dic_disout.values()) # (26,-2)
            # print("最大值和最小值：",max(bbb),min(bbb),bbb[29],bbb[218]) # 26,-2,0,0
            # print("最大值和最小值索引：", bbb.index(max(bbb)), bbb.index(min(bbb))) # 21,639
            # # print(sorted(bbb, reverse=True))
            # has_zero = any(num == 0 for num in bbb)
            # print(has_zero, bbb.count(0))
            # print("---------------------------------") # 104,5,很多0.基本都有值
            # print(differences_sum_dlabel, differences_sum_dout, softmax_lvalues, softmax_dvalues)
            # print(softmax_lvalues[449], softmax_lvalues[639])
            # print(softmax_dvalues[21], softmax_dvalues[639])
            # print("----------------------------------")



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
                loss2 = self.criterion(dis_out.reshape(b,t,n,-1), yl_label.float()) # 鉴别器损失，450个节点的准确优劣分类
                loss3, values_region = static_cal(new_pred, new_label) # 静态公平正则化
                loss4, values_node = dynamic_cal5(dis_labels, dis_out, sample_map, district_nodes, self.district13_road_index)
            
                loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'
                self._logger.info(loss_message.format(epoch + 1, batch_count+1, loss1, loss2, loss3, loss4))

                e_loss.append(loss1.item())
                e_mape.append(mape)
                e_rmse.append(rmse)
                estatic_fair.append(loss3.item())
                edynamic_fair.append(loss4.item())

                sorted_dict = pro_sample2(values_region, values_node, self.district13_road_index)
                sample_list = resampel(sorted_dict,self.sample_num,self.district13_road_index)
                sample_map = sum_map(sample_list, self.sample_num)
                sample_dict = sample_map_district(sample_list, self.district13_road_index)
                district_nodes = get_district_nodes(sample_dict, sample_map)
            


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


