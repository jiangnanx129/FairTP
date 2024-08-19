import torch
from src.base.engine import BaseEngine
# from src.utils.metrics import masked_mape, masked_rmse
from src.utils.metrics import masked_mape, masked_rmse, masked_mpe, masked_mae, masked_mae_region
# from src.utils.metrics_region import masked_mae2

import time
import numpy as np
# from src.engines.IDF_baselines import *
from src.engines.sample_T_single import *
from src.engines.sample_optimal_greedy_T import *  # 先区域后节点
import pickle
from src.utils.graph_algo import normalize_adj_mx, calculate_cheb_poly

'''
用的是astgcn_fair3_T的模型
'''


def get_supports_list(sample_list, adj_mx, adj_type, device):
        
    new_adj = adj_mx[sample_list] # 8k*8k, select 716*8k
    new_adj = new_adj[:,sample_list]
    
    new_adj = normalize_adj_mx(new_adj, adj_type)
    supports = [torch.tensor(i).to(device) for i in new_adj]
    return supports


class GWNET_Engine(BaseEngine):
    def __init__(self, adj_type, adj_mx, T_dynamic, sample_num, district13_road_index, **args):
        super(GWNET_Engine, self).__init__(**args)
        for p in self.model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)
        self.T_dynamic = T_dynamic
        self.sample_num = sample_num
        self.district13_road_index = district13_road_index
        self.adj_mx = adj_mx # (938,938)
        self.adj_type = adj_type
        


    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = 'final_model_s{}_HKALL_1.pt'.format(self._seed)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))
    
    def load_model(self, save_path):
        filename = 'final_model_s{}_HKALL_1.pt'.format(self._seed)
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename)))




    def train_batch(self, district_nodes, epoch):
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

            X, label = self._to_device(self._to_tensor([X, label]))
            b,t,n,c = X.shape # n是采样的数目

            # def forward(self, x, adj_list, label=None):
            pred = self.model(X, label)
            pred, label = self._inverse_transform([pred, label])
            

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
            # mpe2 = masked_mpe(pred, label, mask_value_single).item()
            # mae2 = masked_mae(pred, label, mask_value_single).item()

            # 区域级mae
            loss = self._loss_fn(new_pred, new_label, mask_value)
            mape = masked_mape(new_pred, new_label, mask_value).item()
            rmse = masked_rmse(new_pred, new_label, mask_value).item()

            loss3 = static_cal(new_pred, new_label, self._device) # RSF静态公平正则化
            
            loss4 = -1
            # loss4 = dynamic_cal(pred, label, mask_value_single, yl_values, sample_map, district_nodes) # (b,t,n,co)
            # dynamic_fair.append(loss4.item())
                
            
            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss.item()) # 区域mae
            train_mape2.append(mape2)
            train_mape.append(mape)
            train_rmse.append(rmse)
            static_fair.append(loss3.item()) # 静态损失
            

            loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, node mape:{:.4f}, Loss(mae): {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'
            self._logger.info(loss_message.format(epoch + 1, batch_count+1, mape2, loss, loss3, loss4))

            self._iter_cnt += 1
            batch_count += 1
        return np.mean(train_loss), np.mean(train_mape2), np.mean(train_mape), np.mean(train_rmse), np.mean(static_fair)



    def train(self, district_nodes):
        self._logger.info('Start training!')

        yl_list_train, yl_list_train_node = [],[]
        yl_list_val, yl_list_val_node = [],[]
        yl_list_train2, yl_list_val2 = [],[]

        wait = 0
        min_loss = np.inf
        for epoch in range(self._max_epochs):
            t1 = time.time()
            mtrain_loss, mtrain_mape2, mtrain_mape, mtrain_rmse, mtrain_sfair = self.train_batch(district_nodes, epoch)
            t2 = time.time()

            v1 = time.time()
            mvalid_loss, mvalid_mape2, mvalid_mape, mvalid_rmse, mvalid_sfair = self.evaluate('val', district_nodes, epoch)
            v2 = time.time()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            
            message = 'Epoch: {:03d}, Train, node mape: {:.4f}, Loss(MAE): {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}, SFair: {:.4f}, Time: {:.4f}s, LR: {:.4e}, D_LR: {:.4e}' # , DFair: {:.4f}'
            self._logger.info(message.format(epoch + 1, mtrain_mape2, mtrain_loss, mtrain_mape, mtrain_rmse, mtrain_sfair, (t2 - t1), (v2 - v1), cur_lr))

            message = 'Epoch: {:03d}, Test, node mape: {:.4f}, Loss(MAE): {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}, SFair: {:.4f}, Time: {:.4f}s, LR: {:.4e}, D_LR: {:.4e}' # , DFair: {:.4f}'
            self._logger.info(message.format(epoch + 1, mvalid_mape2, mvalid_loss, mvalid_mape, mvalid_rmse, mvalid_sfair, (t2 - t1), (v2 - v1), cur_lr))


            # 保存数据作baseline，yl_label
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
        # 用这个！计算的是mape！
        filename1 = 'ylvalue_s{}_HKALL_1.pkl'.format(self._seed)
        with open(os.path.join(self._save_path, filename1), 'wb') as f:
            pickle.dump((yl_list_train2, yl_list_val2), f)


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
        e_mape2, e_mpe2, e_mae2 = [],[],[]
        estatic_fair_mae = []
        loss_region_list = []
        
        estatic_fair, edynamic_fair =[],[]
        batch_count = 0
        with torch.no_grad():
            for X, label in self._dataloader[mode + '_loader'].get_iterator():

                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
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
                loss_region = masked_mae_region(new_pred, new_label, mask_value)


                mask_value_single = torch.tensor(0)
                if label.min() < 1:
                    mask_value_single = label.min()
                if self._iter_cnt == 0:
                    print('check single mask value', mask_value_single)

                mape2 = masked_mape(pred, label, mask_value_single).item()
                

                loss = self._loss_fn(new_pred, new_label, mask_value)
                mape = masked_mape(new_pred, new_label, mask_value).item()
                rmse = masked_rmse(new_pred, new_label, mask_value).item()
                loss3 = static_cal(new_pred, new_label, self._device)
                
                loss4 = -1
                # loss4 = dynamic_cal(pred, label, mask_value_single, yl_values, sample_map, district_nodes) # (b,t,n,co)
                # loss4 = dynamic_cal(pred, label, mask_value_single, yl_values, sample_map, district_nodes) # (b,t,n,co)
                # edynamic_fair.append(loss4.item())
                    
                
                e_loss.append(loss.item()) # 区域级mae
                e_mape2.append(mape2)
                e_mape.append(mape)
                e_rmse.append(rmse)
                estatic_fair.append(loss3.item())
                

                loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, node mape: {:.4f}, Loss(mae): {:.4f}, loss3: {:.4f}, loss4: {:.4f}'
                self._logger.info(loss_message.format(epoch + 1, batch_count+1, mape2, loss, loss3, loss4))


                if mode == "test":
                    loss_region_list.append(loss_region)

                batch_count += 1


        if mode == 'val':
            mvalid_loss, mvalid_mape2, mvalid_mape, mvalid_rmse, mvalid_sfair = \
            np.mean(e_loss), np.mean(e_mape2), np.mean(e_mape), np.mean(e_rmse), np.mean(estatic_fair)
            return mvalid_loss, mvalid_mape2, mvalid_mape, mvalid_rmse, mvalid_sfair


        elif mode == 'test':
    
            log = 'Average Test, node mape: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}, SFair: {:.4f}.'
            self._logger.info(log.format(np.mean(e_mape2), np.mean(e_loss), np.mean(e_rmse), np.mean(e_mape), np.mean(estatic_fair)))


            loss_region_tensor = torch.stack(loss_region_list,dim=0)
            print(loss_region_tensor.shape)
            mean_values = loss_region_tensor.mean(dim=0)
            print(mean_values)
            self._logger.info(mean_values)