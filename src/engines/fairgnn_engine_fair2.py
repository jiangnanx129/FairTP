import torch
import numpy as np
from src.base.engine import BaseEngine
from src.utils.metrics import masked_mape, masked_rmse
import time
import os
from src.engines.IDF_baselines import *
from src.engines.sample_T_single import *
from src.utils.metrics_region import masked_mae2




'''算组公平:只求mape而不是mape的差'''
def static_gfair2(pred, label):
    # data: (b,t,n_r,c)
    b,t,n_r,c = pred.shape
    pred = pred.reshape(b*t, -1) # （b*t，n_region）
    label = label.reshape(b*t, -1)

    mape_diff = []

    for i in range(pred.shape[1]): # 区域数量-1-->13-1=12 # i = 0123456789--11
        mape_i = torch.abs(pred[:, i] - label[:, i]) / label[:, i] # 区域i的mape
        mape_diff.append(torch.sum(mape_i)) # mean

    mape_diff_mean = torch.mean(torch.stack(mape_diff), dim=0)

    return mape_diff_mean
    
 


        


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
            loss2 = static_gfair2(new_pred, new_label)
            
            out_static_loss, values_region = static_cal(new_pred, new_label, self._device) # 静态公平正则化


            loss = loss1 + self.a_loss2*loss2 

            loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, Loss: {:.4f}, Loss2: {:.4f}, out static Loss: {:.4f}'
            self._logger.info(loss_message.format(epoch + 1, batch_count+1, loss, loss2, out_static_loss))


            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()
           

            train_loss.append(loss1.item()) # MAE
            train_mape.append(mape)
            train_rmse.append(rmse)
            static_gfairl.append(loss2)
        
            static_ourfairl.append(out_static_loss.item())


            self._iter_cnt += 1
            batch_count += 1
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse), np.mean(static_gfairl), np.mean(static_ourfairl)

    






    def train(self, district_nodes):
        self._logger.info('Start training!')

        wait = 0
        min_loss = np.inf
        for epoch in range(self._max_epochs):
            t1 = time.time()
            mtrain_loss, mtrain_mape, mtrain_rmse, mtrain_gfair, mtrain_ourfair = self.train_batch(district_nodes, epoch)
            t2 = time.time()

            v1 = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse, mvalid_gfair, mvalid_ourfair = self.evaluate('val', district_nodes, epoch)
            v2 = time.time()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            message = 'Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train SGFair: {:.4f}, Train SOURFair: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Valid SGFair: {:.4f}, Valid SOURFair: {:.4f}, Train Time: {:.4f}s/epoch, Valid Time: {:.4f}s, LR: {:.4e}'
            self._logger.info(message.format(epoch + 1, mtrain_loss, mtrain_rmse, mtrain_mape, mtrain_gfair, mtrain_ourfair, \
                                             mvalid_loss, mvalid_rmse, mvalid_mape, mvalid_gfair, mvalid_ourfair, \
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
                loss2 = static_gfair2(new_pred, new_label)
                
                loss = loss1 + self.a_loss2*loss2

                '''看13个区域的情况'''
                loss_region = masked_mae2(new_pred, new_label, mask_value)

            
                out_static_loss, values_region = static_cal(new_pred, new_label, self._device) # 静态公平正则化

                loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, Loss: {:.4f}, Loss2: {:.4f}, out static Loss: {:.4f}'
                self._logger.info(loss_message.format(epoch + 1, batch_count+1, loss, loss2, out_static_loss))


                e_loss.append(loss1.item())
                e_mape.append(mape)
                e_rmse.append(rmse)
                estatic_gfairl.append(loss2)
                
                estatic_ourfairl.append(out_static_loss.item())

                if mode == "test":
                    loss_region_list.append(loss_region)


                batch_count += 1

        if mode == 'val':
            mae, mape, rmse, mvalid_gfair, mvalid_ourfair = \
                np.mean(e_loss), np.mean(e_mape), np.mean(e_rmse), np.mean(estatic_gfairl), np.mean(estatic_ourfairl)
            return mae, mape, rmse, mvalid_gfair, mvalid_ourfair

        elif mode == 'test':
    
            log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test SGFair: {:.4f}, Test SOURFair: {:.4f}'
            self._logger.info(log.format(np.mean(e_loss), np.mean(e_rmse), np.mean(e_mape), np.mean(estatic_gfairl), np.mean(estatic_ourfairl)))

            loss_region_tensor = torch.stack(loss_region_list,dim=0)
            print(loss_region_tensor.shape)
            mean_values = loss_region_tensor.mean(dim=0)
            print(mean_values)
            self._logger.info(mean_values)