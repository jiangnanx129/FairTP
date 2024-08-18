import torch
import numpy as np
from src.base.engine import BaseEngine
from src.utils.metrics import masked_mape, masked_rmse
import time
import os
from src.engines.IDF_baselines import *
from src.engines.sample_T_single import *
from src.utils.metrics_region import masked_mae2


# 希望输出是每个区域一个(b,t,1)
# def static_cal(pred,label): # (b, t, 13, 1), 13表示区域数量，1表示预测的交通值维度
#     b, t = pred.shape[0], pred.shape[1]
#     pred = pred.reshape(b * t, -1) # 将 pred 和 label 转换为形状为 (b*t, 13)
#     label = label.reshape(b * t, -1)

#     # 初始化一个张量用于保存每个区域对之间的 MAPE 差值
#     mape_diff = [] # torch.zeros((78,))

#     # 计算 MAPE 差值
#     idx = 0
#     for i in range(pred.shape[1]-1): # 区域数量-1-->13-1=12 # i = 0123456789--11
#         for j in range(i + 1, pred.shape[1]):
#             mape_i = torch.abs(pred[:, i] - label[:, i]) / label[:, i] # 区域i的mape
#             mape_j = torch.abs(pred[:, j] - label[:, j]) / label[:, j] # 区域j的mape
#             # mape_diff[idx] = torch.mean(mape_i - mape_j)
#             # mape_diff.append(torch.abs(torch.sum(mape_i - mape_j))) # mean换成sum 
#             mape_diff.append(torch.sum(torch.abs(mape_i - mape_j)))
#             idx += 1
#         # print("--------------------:",mape_i, mape_j,mape_j.shape)
        
#     mape_diff_mean = torch.mean(torch.stack(mape_diff), dim=0)

#     return mape_diff_mean


'''算组公平'''
def static_gfair(data):
    # data: (b,t,n_r,c)
    b,t,n_r,c = data.shape
    data = data.reshape(b*t, -1) # （b*t，n_region）
    sum_a, count_a = 0,0
    sum_a_list = []
    for i in range(data.shape[1]-1): # 循环每个区域
        for j in range(i + 1, data.shape[1]):
            # a =torch.abs(data[:, i] - data[:, j]) # torch.Size([b*t])
            # print(a.shape)
            sum_a += torch.mean(torch.abs(data[:, i] - data[:, j])) # torch.abs得到形状为(b*t,)的一维张量
            count_a += 1

    return sum_a, sum_a/count_a
    


        
'''算组中个体公平'''
def static_ifair(data, district_nodes):
    # data: (b,t,n,c)
    sum_a_list, sum_a_list2 = [],[] # 每个元素是某一区域内所有个体的平均差异

    for region, nodes_list in district_nodes.items(): # region是str '0', 单独循环每个区域
        data_a = data[:,:,nodes_list] # 得到(b,t,len(list),c)
        data_a = data_a.reshape(-1, data_a.shape[2]) # (b*t, n_node)
        sum_a, count_a = 0,0

        for i in range(data_a.shape[1]-1): # 循环某个区域的所有节点
            for j in range(i + 1, data_a.shape[1]):
                sum_a += torch.mean(torch.abs(data_a[:, i] - data_a[:, j])) # torch.abs得到形状为(b*t,)的一维张量
                count_a += 1
        
        sum_a_list.append(sum_a/count_a)
        sum_a_list2.append(sum_a)

    # print(sum_a_list)
    # print(torch.stack(sum_a_list).shape) # torch.Size([13])

    return torch.mean(torch.stack(sum_a_list2)), torch.mean(torch.stack(sum_a_list))
            
            
        # new_pred[:, :, int(region)] = data[:, :, nodes_list]
        # new_label[:, :, int(region)] = label[:, :, nodes_list].mean(dim=2)#, keepdim=True) #.mean(dim=2)
            
    return 1


class FAIRGNN_Engine(BaseEngine):
    def __init__(self, a_loss2, a_loss3, **args):
        super(FAIRGNN_Engine, self).__init__(**args)

        self.a_loss2 = a_loss2 
        self.a_loss3 = a_loss3

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = 'final_model_s{}_HK_fair_1.pt'.format(self._seed)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))
    
    def load_model(self, save_path):
        filename = 'final_model_s{}_HK_fair_1.pt'.format(self._seed)
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename)))


    def train_batch(self, district_nodes, epoch):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        static_gfairl, static_ifairl, static_ourfairl = [],[],[]
        loss_ad_list, loss_cov_list = [], []
        batch_count = 0
        self._dataloader['train_loader'].shuffle()
        for X, label in self._dataloader['train_loader'].get_iterator():
            
            self._optimizer.zero_grad()

            X, label = self._to_device(self._to_tensor([X, label]))
            pred = self.model(X, label)
            pred, label = self._inverse_transform([pred, label])


            b,t,n,c = X.shape
            
            new_pred = torch.zeros(b, t, len(list(district_nodes.keys())), c)
            new_label = torch.zeros(b, t, len(list(district_nodes.keys())), c)
            # print("1:", new_pred.shape) # (b,t,region,c)
            # print(new_pred)

            for region, nodes_list in district_nodes.items(): # region是str '0'
                new_pred[:, :, int(region)] = pred[:, :, nodes_list].mean(dim=2)#, keepdim=True)#.mean(dim=2)
                new_label[:, :, int(region)] = label[:, :, nodes_list].mean(dim=2)#, keepdim=True) #.mean(dim=2)
            
            # print("2:", new_pred.shape) # (b,t,region,c)
            # print("3:", pred[:, :, nodes_list].shape) # (b,t,len(node_list),c)
            # print("4:", pred[:, :, nodes_list].mean(dim=2).shape) # (b,t,c)

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

            
            mape = masked_mape(new_pred, new_label, mask_value).item()
            rmse = masked_rmse(new_pred, new_label, mask_value).item()
            
            '''
            计算损失，公平性+准确性
            1. 组间公平
            2. 每个组间的个体公平
            '''
            loss1 = self._loss_fn(new_pred, new_label, mask_value)
            loss2_1, loss2_2 = static_gfair(new_pred)
            # print(loss1, loss2_1, loss2_2) # 大 小
            loss3_1, loss3_2 = static_ifair(pred, district_nodes)
            # print(loss3_1, loss3_2)
            
            out_static_loss, values_region = static_cal(new_pred, new_label, self._device) # 静态公平正则化


            loss = loss1 + self.a_loss2*loss2_1 + self.a_loss3*loss3_2

            loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, Loss: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}, out static Loss: {:.4f}'
            self._logger.info(loss_message.format(epoch + 1, batch_count+1, loss, loss2_1, loss3_2, out_static_loss))


            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()
           

            train_loss.append(loss1.item()) # MAE
            train_mape.append(mape)
            train_rmse.append(rmse)
            static_gfairl.append(loss2_1.item())
            static_ifairl.append(loss3_2.item())
            static_ourfairl.append(out_static_loss.item())


            self._iter_cnt += 1
            batch_count += 1
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse), np.mean(static_gfairl), np.mean(static_ifairl), np.mean(static_ourfairl)

    






    def train(self, district_nodes):
        self._logger.info('Start training!')

        wait = 0
        min_loss = np.inf
        for epoch in range(self._max_epochs):
            t1 = time.time()
            mtrain_loss, mtrain_mape, mtrain_rmse, mtrain_gfair, mtrain_ifair, mtrain_ourfair = self.train_batch(district_nodes, epoch)
            t2 = time.time()

            v1 = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse, mvalid_gfair, mvalid_ifair, mvalid_ourfair = self.evaluate('val', district_nodes, epoch)
            v2 = time.time()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            message = 'Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train SGFair: {:.4f}, Train SIFair: {:.4f}, Train SOURFair: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Valid SGFair: {:.4f}, Valid DIFair: {:.4f}, Valid SOURFair: {:.4f}, Train Time: {:.4f}s/epoch, Valid Time: {:.4f}s, LR: {:.4e}'
            self._logger.info(message.format(epoch + 1, mtrain_loss, mtrain_rmse, mtrain_mape, mtrain_gfair, mtrain_ifair, mtrain_ourfair, \
                                             mvalid_loss, mvalid_rmse, mvalid_mape, mvalid_gfair, mvalid_ifair, mvalid_ourfair, \
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

        self.evaluate('test', district_nodes, epoch)


    def evaluate(self, mode, district_nodes, epoch):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()

        preds = []
        labels = []
        e_loss = []
        e_mape = []
        e_rmse = []
        estatic_gfairl, estatic_ifairl, estatic_ourfairl =[],[],[]
        loss_region_list = []
        
        batch_count = 0
        with torch.no_grad():
            for X, label in self._dataloader[mode + '_loader'].get_iterator():
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                pred = self.model(X, label)
                pred, label = self._inverse_transform([pred, label])

                mask_value_single = torch.tensor(0)
                if label.min() < 1:
                    mask_value_single = label.min()
                if self._iter_cnt == 0:
                    print('check single mask value', mask_value_single)


                b,t,n,c = X.shape
                new_pred = torch.zeros(b, t, len(list(district_nodes.keys())), c)
                new_label = torch.zeros(b, t, len(list(district_nodes.keys())), c)
                for region, nodes_list in district_nodes.items(): # region是str '0' # sample2.py中所有round+0.5确保0-12均有采样，即有0-12
                    new_pred[:, :, int(region)] = pred[:, :, nodes_list].mean(dim=2)#, keepdim=True)#.mean(dim=2)
                    new_label[:, :, int(region)] = label[:, :, nodes_list].mean(dim=2)#, keepdim=True) #.mean(dim=2)

                mask_value = torch.tensor(0)
                if new_label.min() < 1:
                    mask_value = new_label.min()

                
                mape = masked_mape(new_pred, new_label, mask_value).item()
                rmse = masked_rmse(new_pred, new_label, mask_value).item()
                
                loss1 = self._loss_fn(new_pred, new_label, mask_value)
                loss2_1, loss2_2 = static_gfair(new_pred)
                loss3_1, loss3_2 = static_ifair(pred, district_nodes)
                loss = loss1 + self.a_loss2*loss2_1 + self.a_loss3*loss3_2

                '''看13个区域的情况'''
                loss_region = masked_mae2(new_pred, new_label, mask_value)

            
                out_static_loss, values_region = static_cal(new_pred, new_label, self._device) # 静态公平正则化

                loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, Loss: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}, out static Loss: {:.4f}'
                self._logger.info(loss_message.format(epoch + 1, batch_count+1, loss, loss2_1, loss3_2, out_static_loss))


                e_loss.append(loss1.item())
                e_mape.append(mape)
                e_rmse.append(rmse)
                estatic_gfairl.append(loss2_1.item())
                estatic_ifairl.append(loss3_2.item())
                estatic_ourfairl.append(out_static_loss.item())

                if mode == "test":
                    loss_region_list.append(loss_region)


                batch_count += 1

        if mode == 'val':
            mae, mape, rmse, mvalid_gfair, mvalid_ifair, mvalid_ourfair = \
                np.mean(e_loss), np.mean(e_mape), np.mean(e_rmse), np.mean(estatic_gfairl), np.mean(estatic_ifairl), np.mean(estatic_ourfairl)
            return mae, mape, rmse, mvalid_gfair, mvalid_gfair, mvalid_ourfair

        elif mode == 'test':
    
            log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test SGFair: {:.4f}, Test SIFair: {:.4f}, Test SOURFair: {:.4f}'
            self._logger.info(log.format(np.mean(e_loss), np.mean(e_rmse), np.mean(e_mape), np.mean(estatic_gfairl), np.mean(estatic_ifairl), np.mean(estatic_ourfairl)))

            loss_region_tensor = torch.stack(loss_region_list,dim=0)
            print(loss_region_tensor.shape)
            mean_values = loss_region_tensor.mean(dim=0)
            print(mean_values)
            self._logger.info(mean_values)