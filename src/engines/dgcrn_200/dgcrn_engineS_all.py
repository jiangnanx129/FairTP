import torch
from src.base.engine import BaseEngine
# from src.utils.metrics import masked_mape, masked_rmse
from src.utils.metrics import masked_mape, masked_rmse, masked_mpe, masked_mae
from src.utils.metrics_region import masked_mae2

import time
import numpy as np
from src.engines.IDF_baselines import *
from src.engines.sample_optimal_greedy_T import *  # 先区域后节点
import pickle
from src.utils.graph_algo import normalize_adj_mx, calculate_cheb_poly

'''
用的是astgcn_fair3_T的模型
'''

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

def get_sample_adj(sample_list, adj_mx, device):
        
    new_adj = adj_mx[sample_list] # 8k*8k, select 716*8k
    new_adj = new_adj[:,sample_list]

    new_adj = normalize_adj_mx(new_adj, 'doubletransition')
    supports = [torch.tensor(i).to(device) for i in new_adj]

    return supports


class DGCRN_Engine(BaseEngine):
    def __init__(self, step_size, horizon, adj_mx, sam_num, T_dynamic, sample_num, district13_road_index, **args):
        super(DGCRN_Engine, self).__init__(**args)
        
        self.T_dynamic = T_dynamic
        self.sample_num = sample_num
        self.district13_road_index = district13_road_index
        self.adj_mx = adj_mx # (938,938)

        self.sam_num = sam_num

        self._step_size = step_size
        self._horizon = horizon
        self._task_level = 0
        


    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = 'final_model_s{}_HK_3_1e-4_S_b42_T12_all.pt'.format(self._seed)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))
    
    def load_model(self, save_path):
        filename = 'final_model_s{}_HK_3_1e-4_S_b42_T12_all.pt'.format(self._seed)
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename)))




    def train_batch(self, sample_list, time_T, yl_values, sample_map, district_nodes, epoch):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        train_mape2, train_mpe2, train_mae2 = [],[],[]
        static_fair_mae = []

        static_fair, dynamic_fair = [],[]
        batch_count = 0
        self._dataloader['train_loader'].shuffle()
        for X, label in self._dataloader['train_loader'].get_iterator():
            self._optimizer.zero_grad()

            # if self._iter_cnt % self._step_size == 0 and self._task_level < self._horizon:
            #     self._task_level += 1

            self._task_level = 12

            # X (b, t, n, f), label (b, t, n, 1)
            X, label = self._to_device(self._to_tensor([X, label]))
            '''新加的采样模块，不是直接读取采样后的数据，而是采样'''
            b,t,n,c = X.shape # n是采样的数目

            pred = self.model(X, label, self._iter_cnt, self._task_level)
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

            mask_value_single = torch.tensor(0)
            if label.min() < 1:
                mask_value_single = label.min()
            if self._iter_cnt == 0:
                print('check single mask value', mask_value_single)


            # 节点级别的mape
            mape2 = masked_mape(pred, label, mask_value_single).item()
            mpe2 = masked_mpe(pred, label, mask_value_single).item()
            mae2 = masked_mae(pred, label, mask_value_single).item()


            loss = self._loss_fn(new_pred, new_label, mask_value)
            mape = masked_mape(new_pred, new_label, mask_value).item()
            rmse = masked_rmse(new_pred, new_label, mask_value).item()
            # loss3 = static_cal(new_pred,new_label) # 静态公平正则化
            # loss4 = dynamic_cal(pred, label, mask_value_single, yl_values, sample_map, district_nodes) # (b,t,n,co)
            
            # loss3_mae = static_cal2_mae(new_pred, new_label, mask_value)
            # loss3_mape = static_cal2_mape(new_pred, new_label, mask_value)
            lpe, loss3_mape, loss3_mae = static_cal(new_pred,new_label) # 静态公平正则化
            loss4 = dynamic_cal(pred, label, mask_value_single, yl_values, sample_map, district_nodes) # (b,t,n,co)
            dynamic_fair.append(loss4.item())
                
            
            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss.item())
            train_mape.append(mape)
            train_mape2.append(mape2)
            train_mpe2.append(mpe2)
            train_mae2.append(mae2)

            train_rmse.append(rmse)
            static_fair.append(loss3_mae.item()) # mpe
            static_fair_mae.append(loss3_mape.item()) # mae

            loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, mape2:{:.4f}, Loss1: {:.4f}, Loss3_mae: {:.4f}, Loss3_mape: {:.4f}, Loss4: {:.4f}'
            self._logger.info(loss_message.format(epoch + 1, batch_count+1, mape2, loss, loss3_mae, loss3_mape, loss4))

            self._iter_cnt += 1
            batch_count += 1
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_mape2), np.mean(train_mpe2), np.mean(train_mae2), np.mean(train_rmse), np.mean(static_fair), np.mean(dynamic_fair), np.mean(static_fair_mae)



    def train(self, sample_list, time_T, yl_values, sample_map, district_nodes):
        self._logger.info('Start training!')

        yl_list_train, yl_list_train_node = [],[]
        yl_list_val, yl_list_val_node = [],[]
        yl_list_train2, yl_list_val2 = [],[]

        wait = 0
        min_loss = np.inf
        for epoch in range(self._max_epochs):
            t1 = time.time()
            mtrain_loss, mtrain_mape, mtrain_mape2, mtrain_mpe2, mtrain_mae2, mtrain_rmse, mtrain_sfair, mtrain_dfair, mtrain_sfair_mae = self.train_batch(sample_list, time_T, yl_values, sample_map, district_nodes, epoch)
            t2 = time.time()

            v1 = time.time()
            mvalid_loss, mvalid_mape, mvalid_mape2, mvalid_mpe2, mvalid_mae2, mvalid_rmse, mvalid_sfair, mvalid_dfair, mvalid_sfair_mae = self.evaluate('val', sample_list, time_T, yl_values, sample_map, district_nodes, epoch)
            v2 = time.time()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            # message = 'Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train SFair: {:.4f}, Train DFair: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Valid SFair: {:.4f}, Valid DFair: {:.4f}, Train Time: {:.4f}s/epoch, Valid Time: {:.4f}s, LR: {:.4e}'
            # self._logger.info(message.format(epoch + 1, mtrain_loss, mtrain_rmse, mtrain_mape, mtrain_sfair, mtrain_dfair, \
            #                                  mvalid_loss, mvalid_rmse, mvalid_mape, mvalid_sfair, mvalid_dfair, \
            #                                  (t2 - t1), (v2 - v1), cur_lr))

            message = 'Epoch: {:03d}, Train mape2: {:.4f}, Train mpe2: {:.4f}, Train mae2: {:.4f}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train SFair_mae: {:.4f}, Train DFair: {:.4f}, Train SFair_mape: {:.4f}, Valid mape2: {:.4f}, Valid mpe2: {:.4f}, Valid mae2: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Valid SFair: {:.4f}, Valid DFair: {:.4f}, Valid SFair_mae: {:.4f}, Train Time: {:.4f}s/epoch, Valid Time: {:.4f}s, LR: {:.4e}'
            self._logger.info(message.format(epoch + 1, mtrain_mape2, mtrain_mpe2, mtrain_mae2, mtrain_loss, mtrain_rmse, mtrain_mape, mtrain_sfair, mtrain_dfair, mtrain_sfair_mae,\
                                             mvalid_mape2, mvalid_mpe2, mvalid_mae2, mvalid_loss, mvalid_rmse, mvalid_mape, mvalid_sfair, mvalid_dfair, mvalid_sfair_mae,\
                                             (t2 - t1), (v2 - v1), cur_lr))

            yl_list_train.append(mtrain_mpe2.item())
            yl_list_val.append(mvalid_mpe2.item())
            yl_list_train_node.append(mtrain_mae2.item())
            yl_list_val_node.append(mvalid_mae2.item())
            yl_list_train2.append(mtrain_mape2.item())
            yl_list_val2.append(mvalid_mape2.item())

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
        

        # 将两个列表存储到文件中
        filename1 = 'ylvalue_s{}_HK_3_1e-4_node_mpe2_S_b42_T12_all.pkl'.format(self._seed)
        with open(os.path.join(self._save_path, filename1), 'wb') as f:
            pickle.dump((yl_list_train, yl_list_val), f)

        filename2 = 'ylvalue_s{}_HK_3_1e-4_node_mae2_S_b42_T12_all.pkl'.format(self._seed)
        with open(os.path.join(self._save_path, filename2), 'wb') as f:
            pickle.dump((yl_list_train_node, yl_list_val_node), f)

        # 用这个！计算的是mape！
        filename1 = 'ylvalue_s{}_HK_3_1e-4_node_mape2_S_b42_T12_all.pkl'.format(self._seed)
        with open(os.path.join(self._save_path, filename1), 'wb') as f:
            pickle.dump((yl_list_train2, yl_list_val2), f)


        self.evaluate('test', sample_list, time_T, yl_values, sample_map, district_nodes, epoch)



    def evaluate(self, mode, sample_list, time_T, yl_values, sample_map, district_nodes, epoch):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()

        preds = []
        labels = []
        e_loss = []
        e_mape = []
        e_rmse = []
        e_mape2, e_mpe2, e_mae2 = [],[],[]
        estatic_fair_mae = []
        loss_region_list = []
        
        estatic_fair, edynamic_fair =[],[]
        batch_count = 0
        with torch.no_grad():
            for X, label in self._dataloader[mode + '_loader'].get_iterator():
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))

            
                self._task_level=12
                
                pred = self.model(X, label)
                pred, label = self._inverse_transform([pred, label])
            
                b,t,n,c = X.shape # n是采样的数目,450


                new_pred = torch.zeros(b, t, len(list(district_nodes.keys())), c)
                new_label = torch.zeros(b, t, len(list(district_nodes.keys())), c)
                for region, nodes_list in district_nodes.items(): # region是str '0' # sample2.py中所有round+0.5确保0-12均有采样，即有0-12
                    new_pred[:, :, int(region)] = pred[:, :, nodes_list].mean(dim=2)#, keepdim=True)#.mean(dim=2)
                    new_label[:, :, int(region)] = label[:, :, nodes_list].mean(dim=2)#, keepdim=True) #.mean(dim=2)

                mask_value = torch.tensor(0)
                if new_label.min() < 1:
                    mask_value = new_label.min()

                '''看13个区域的情况'''
                loss_region = masked_mae2(new_pred, new_label, mask_value)


                mask_value_single = torch.tensor(0)
                if label.min() < 1:
                    mask_value_single = label.min()
                if self._iter_cnt == 0:
                    print('check single mask value', mask_value_single)

                mape2 = masked_mape(pred, label, mask_value_single).item()
                mpe2 = masked_mpe(pred, label, mask_value_single).item()
                mae2 = masked_mae(pred, label, mask_value_single).item()

                loss = self._loss_fn(new_pred, new_label, mask_value)
                mape = masked_mape(new_pred, new_label, mask_value).item()
                rmse = masked_rmse(new_pred, new_label, mask_value).item()
                # loss3 = static_cal(new_pred,new_label) # 静态公平正则化
                # loss4 = dynamic_cal(pred, label, mask_value_single, yl_values, sample_map, district_nodes) # (b,t,n,co)
                lpe, loss3_mape, loss3_mae = static_cal(new_pred,new_label) # 静态公平正则化
                # loss4 = dynamic_cal(pred, label, mask_value_single, yl_values, sample_map, district_nodes) # (b,t,n,co)
                loss4 = dynamic_cal(pred, label, mask_value_single, yl_values, sample_map, district_nodes) # (b,t,n,co)
                edynamic_fair.append(loss4.item())
                    

                # loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, Loss1: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'
                # self._logger.info(loss_message.format(epoch + 1, batch_count+1, loss, loss3, loss4))
                
                e_loss.append(loss.item())
                e_mape.append(mape)
                e_mape2.append(mape2)
                e_mpe2.append(mpe2)
                e_mae2.append(mae2)

                e_rmse.append(rmse)
                estatic_fair.append(loss3_mae.item())
                estatic_fair_mae.append(loss3_mape.item())

                loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, mape2: {:.4f}, mpe2: {:.4f}, mae2: {:.4f}, Loss1: {:.4f}, Loss3_mae: {:.4f}, Loss3_mape: {:.4f}, Loss4: {:.4f}'
                self._logger.info(loss_message.format(epoch + 1, batch_count+1, mape2, mpe2, mae2, loss, loss3_mae, loss3_mape, loss4))


                if mode == "test":
                    loss_region_list.append(loss_region)

                batch_count += 1

        if mode == 'val':
            # mae, mape, rmse, mvalid_sfair, mvalid_dfair= np.mean(e_loss), np.mean(e_mape), np.mean(e_rmse), np.mean(estatic_fair), np.mean(edynamic_fair)
            # return mae, mape, rmse, mvalid_sfair, mvalid_dfair
            mae, mape, mape2, mpe2, mae2, rmse, mvalid_sfair, mvalid_dfair, mvalid_sfair_mae= np.mean(e_loss), np.mean(e_mape), np.mean(e_mape2), np.mean(e_mpe2), np.mean(e_mae2), np.mean(e_rmse), np.mean(estatic_fair), np.mean(edynamic_fair), np.mean(estatic_fair_mae)
            return mae, mape, mape2, mpe2, mae2, rmse, mvalid_sfair, mvalid_dfair, mvalid_sfair_mae

        elif mode == 'test':
    
            # log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test SFair: {:.4f}, Test DFair: {:.4f}'
            # self._logger.info(log.format(np.mean(e_loss), np.mean(e_rmse), np.mean(e_mape), np.mean(estatic_fair), np.mean(edynamic_fair)))
            log = 'Average Test mape2: {:.4f}, mpe2: {:.4f}, mae2: {:.4f}, MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test SFair_mae: {:.4f}, Test DFair: {:.4f}, Test SFair_mape: {:.4f}'
            self._logger.info(log.format(np.mean(e_mape2), np.mean(e_mpe2), np.mean(e_mae2), np.mean(e_loss), np.mean(e_rmse), np.mean(e_mape), np.mean(estatic_fair), np.mean(edynamic_fair), np.mean(estatic_fair_mae)))

            loss_region_tensor = torch.stack(loss_region_list,dim=0)
            print(loss_region_tensor.shape)
            mean_values = loss_region_tensor.mean(dim=0)
            print(mean_values)
            self._logger.info(mean_values)