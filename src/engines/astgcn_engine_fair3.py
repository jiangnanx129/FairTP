import torch
from src.base.engine import BaseEngine
from src.utils.metrics import masked_mape, masked_rmse
import time
import numpy as np
from src.engines.IDF_baselines import *
from src.engines.sample_optimal_xhbl import * # 循环遍历实现优化采样

# 希望输出是每个区域一个(b,t,1)
def static_cal(pred,label): # (b, t, 13, 1), 13表示区域数量，1表示预测的交通值维度
    b, t = pred.shape[0], pred.shape[1]
    pred = pred.reshape(b * t, -1) # 将 pred 和 label 转换为形状为 (b*t, 13)
    label = label.reshape(b * t, -1)

    # 初始化一个张量用于保存每个区域对之间的 MAPE 差值
    mape_diff = [] # torch.zeros((78,))

    # 计算 MAPE 差值
    idx = 0
    for i in range(pred.shape[1]-1): # 区域数量-1-->13-1=12 # i = 0123456789--11
        mape_i = torch.abs(pred[:, i] - label[:, i]) / label[:, i] # 区域i的mape
        for j in range(i + 1, pred.shape[1]):
            
            mape_j = torch.abs(pred[:, j] - label[:, j]) / label[:, j] # 区域j的mape
            # mape_diff[idx] = torch.mean(mape_i - mape_j)
            # mape_diff.append(torch.abs(torch.sum(mape_i - mape_j))) # mean换成sum 
            mape_diff.append(torch.sum(torch.abs(mape_i - mape_j)))
            idx += 1
        # print("--------------------:",mape_i, mape_j,mape_j.shape)
        
    mape_diff_mean = torch.mean(torch.stack(mape_diff), dim=0)

    return mape_diff_mean

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
    def __init__(self, criterion, d_lrate, n_number,model_params, D_params, optimizer_D, scheduler_D, sample_num, district13_road_index, **args):
        super(ASTGCN_Engine, self).__init__(**args)
        self.criterion = criterion
        # self.T_dynamic = T_dynamic
        self.sample_num = sample_num
        self.district13_road_index = district13_road_index
        self.d_lrate = d_lrate
        self.n_number = n_number # 从n_number个组合中选出最合适的采样
        self.model_params = model_params # encoder+decoder的参数
        self.D_params = D_params # discriminator的参数
        self.optimizer_D = optimizer_D # BaseEngine有self._lr_scheduler = scheduler
        self.scheduler_D = scheduler_D # BaseEngine有self._lr_scheduler = scheduler

        for p in self.model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = 'final_model_s{}_HK_fair1.pt'.format(self._seed)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))
    
    def load_model(self, save_path):
        filename = 'final_model_s{}_HK_fair1.pt'.format(self._seed)
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename)))

    def train_batch(self, sample_list, sample_map, sample_dict, district_nodes, yl_values, epoch, total_nodes):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        dis_loss, static_fair, dynamic_fair4, dynamic_fair44 = [],[]
        batch_count = 0
        node_count_global = {}

        self._dataloader['train_loader'].shuffle()
        for X, label in self._dataloader['train_loader'].get_iterator():

            self._optimizer.zero_grad() # encoder和decoder
            self.optimizer_D.zero_grad() # 鉴别器优化器

            # X (b, t, n, f), label (b, t, n, 1)
            X, label = self._to_device(self._to_tensor([X, label]))
            pred, hidd_state, dis_out  = self.model(X, label)
            pred, label = self._inverse_transform([pred, label])
            
            mask_value_single = torch.tensor(0)
            if label.min() < 1:
                mask_value_single = label.min()
            if self._iter_cnt == 0:
                print('check single mask value', mask_value_single)

            # 基于模型输出pred和label的比较计算mape，mape和yl_value比较得到鉴别器的yl_label(0/1)
            mape_data = calcu_mape2(pred, label, mask_value_single)
            yl_label = give_yl_label2(mape_data, yl_values)

            b,t,n,c = X.shape
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

            loss1 = self._loss_fn(new_pred, new_label, mask_value)
            mape = masked_mape(new_pred, new_label, mask_value).item()
            rmse = masked_rmse(new_pred, new_label, mask_value).item()
            loss2 = self.criterion(dis_out, yl_label.float()) # 鉴别器损失，450个节点的准确优劣分类
            loss3 = static_cal(new_pred,new_label) # 静态公平正则化
            loss4, loss44, node_count_global = dynamic_cal5_global3(dis_out, sample_map, district_nodes, self.district13_road_index, node_count_global) # 更新了node_count_global
            # loss4 = dynamic_cal(pred, label, mask_value_single, yl_values, sample_map, district_nodes) # (b,t,n,co)
    
            loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, Loss1: {:.4f}, Dis Loss: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}, Loss44: {:.4f}'
            self._logger.info(loss_message.format(epoch + 1, batch_count+1, loss1, loss2, loss3, loss4, loss44))

            loss = loss1 + 0.01*loss3 + 0.5*loss4 

            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model_params, self._clip_grad_value)
            self._optimizer.step()

            loss2.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.D_params, self._clip_grad_value)
            self.optimizer_D.step()

            train_loss.append(loss1.item())
            train_mape.append(mape)
            train_rmse.append(rmse)
            dis_loss.append(loss2.item())
            static_fair.append(loss3.item())
            dynamic_fair4.append(loss4.item())
            dynamic_fair44.append(loss44.item())

            '''重新采样部分'''
            sample_list = optimize_selection(self.district13_road_index, total_nodes, self.sample_num, node_count_global, self.n_number)
            sample_map = sum_map(sample_list, self.sample_num)
            sample_dict = sample_map_district(sample_list, self.district13_road_index)
            district_nodes = get_district_nodes(sample_dict, sample_map)
            

            self._iter_cnt += 1
            batch_count += 1
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse), np.mean(dis_loss), np.mean(static_fair), np.mean(dynamic_fair4), np.mean(dynamic_fair44)



    def train(self, sample_list, sample_map, sample_dict, district_nodes, yl_values):
        self._logger.info('Start training!')

        wait = 0
        min_loss = np.inf
        for epoch in range(self._max_epochs):
            t1 = time.time()
            mtrain_loss, mtrain_mape, mtrain_rmse, mtraind_loss, mtrain_sfair, mtrain_dfair4, mtrain_dfair44 = self.train_batch(yl_values, sample_map, district_nodes, epoch)
            t2 = time.time()












            v1 = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse, mvalid_sfair, mvalid_dfair = self.evaluate('val', yl_values, sample_map, district_nodes, epoch)
            v2 = time.time()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

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

        self.evaluate('test', yl_values, sample_map, district_nodes, epoch)



    def evaluate(self, mode, yl_values, sample_map, district_nodes, epoch):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()

        preds = []
        labels = []
        e_loss = []
        e_mape = []
        e_rmse = []
        estatic_fair, edynamic_fair =[],[]
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

                loss = self._loss_fn(new_pred, new_label, mask_value)
                mape = masked_mape(new_pred, new_label, mask_value).item()
                rmse = masked_rmse(new_pred, new_label, mask_value).item()
                loss3 = static_cal(new_pred,new_label) # 静态公平正则化
                loss4 = dynamic_cal(pred, label, mask_value_single, yl_values, sample_map, district_nodes) # (b,t,n,co)
    
                loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, Loss1: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'
                self._logger.info(loss_message.format(epoch + 1, batch_count+1, loss, loss3, loss4))


                e_loss.append(loss.item())
                e_mape.append(mape)
                e_rmse.append(rmse)
                estatic_fair.append(loss3.item())
                edynamic_fair.append(loss4.item())
                batch_count += 1

        if mode == 'val':
            mae, mape, rmse, mvalid_sfair, mvalid_dfair= np.mean(e_loss), np.mean(e_mape), np.mean(e_rmse), np.mean(estatic_fair), np.mean(edynamic_fair)
            return mae, mape, rmse, mvalid_sfair, mvalid_dfair

        elif mode == 'test':
    
            log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test SFair: {:.4f}, Test DFair: {:.4f}'
            self._logger.info(log.format(np.mean(e_loss), np.mean(e_rmse), np.mean(e_mape), np.mean(estatic_fair), np.mean(edynamic_fair)))