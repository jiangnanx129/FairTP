import torch
from src.base.engine import BaseEngine
from src.utils.metrics import masked_mape, masked_rmse
import time
import numpy as np
# from src.engines.IDF_baselines import *
from src.engines.sample import *
# from src.engines.sample_optimal_xhbl import * # 循环遍历实现优化采样
# from src.engines.sample_optimal_greedy_test import * # 循环遍历实现优化采样
from src.engines.sample_optimal_greedy_test2 import *  # 先区域后节点
'''两个优化器分先后'''

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


class AGCRN_Engine(BaseEngine):
    def __init__(self, criterion, d_lrate, n_number,model_params, D_params, optimizer_D, scheduler_D, a_loss3, a_loss4, T_dynamic, sample_num, district13_road_index, **args):
        super(AGCRN_Engine, self).__init__(**args)
        '''参数初始化方式'''
        # for p in self.model.parameters():
        #     if p.dim() > 1:
        #         torch.nn.init.xavier_uniform_(p)
        #     else:
        #         torch.nn.init.uniform_(p)
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
        filename = 'final_model_s{}_HK3_disout4_4.pt'.format(self._seed)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))
    
    def load_model(self, save_path):
        filename = 'final_model_s{}_HK3_disout4_4.pt'.format(self._seed)
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename)))

    def train_batch(self, sample_list, sample_map, sample_dict, district_nodes, yl_values, epoch, total_nodes):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        dis_loss, static_fair, dynamic_fair =[],[],[]
        self._dataloader['train_loader'].shuffle()
        batch_count = 0
        node_count_global = {} # 全局，贯穿1个epoch内所有batch记录采样过的个体情况，采样次数
        node_count_global = calcute_global_dic(node_count_global, sample_list)

        for X, label in self._dataloader['train_loader'].get_iterator():
            
            # X (b, t, n, f), label (b, t, n, 1)
            X, label = self._to_device(self._to_tensor([X, label]))
            X, label= X[:, :, sample_list, :], label[:, :, sample_list, :] # (b,t,938,ci)--->(b,t,450,ci)
            b,t,n,c = X.shape # n是采样的数目
        

            self.model.classif.requires_grad_(False)
            self.model.node_embed.requires_grad_(True)
            self.model.encoder.requires_grad_(True)
            self.model.end_conv.requires_grad_(True)
            self._optimizer.zero_grad()

            pred, dis_out = self.model(X, label)
            dis_labels = torch.where(dis_out > 0.5, torch.ones_like(dis_out), torch.full_like(dis_out, -1)).reshape(b,t, n, -1) # 使得dis_label具有梯度 .requires_grad_()
            pred, label1 = self._inverse_transform([pred, label])

            '''后面算'''
            # mape_data = calcu_mape2(pred, label1, mask_value_single)
            # yl_label = give_yl_label2(mape_data, yl_values)

            new_pred = torch.zeros(b, t, len(list(district_nodes.keys())), c)
            new_label = torch.zeros(b, t, len(list(district_nodes.keys())), c)
            for region, nodes_list in district_nodes.items(): # region是str '0' # sample2.py中所有round+0.5确保0-12均有采样，即有0-12
                new_pred[:, :, int(region)] = pred[:, :, nodes_list].mean(dim=2)#, keepdim=True)#.mean(dim=2)
                new_label[:, :, int(region)] = label1[:, :, nodes_list].mean(dim=2)#, keepdim=True) #.mean(dim=2)
            
            # 针对13个区域
            mask_value = torch.tensor(0)
            if new_label.min() < 1:
                mask_value = new_label.min()
            if self._iter_cnt == 0:
                print('check mask value', mask_value)

            loss1 = self._loss_fn(new_pred, new_label, mask_value)
            mape = masked_mape(new_pred, new_label, mask_value).item()
            rmse = masked_rmse(new_pred, new_label, mask_value).item()
            # loss2_1 = self.criterion(dis_out, yl_label.float()) # 鉴别器损失，450个节点的准确优劣分类
            
            loss3, values_region = static_cal(new_pred, new_label) # 静态公平正则化
            loss4 = dynamic_cal5_global(dis_labels, dis_out, sample_map, district_nodes) 
            # loss4, node_count_global = dynamic_cal5_global_xhbl(dis_labels, dis_out, sample_map, district_nodes, self.district13_road_index, node_count_global) # 更新了node_count_global
            
            loss = loss1 + self.a_loss3*loss3 + self.a_loss4*loss4  # HK 0.1, dis_lobel 0.01
            loss.backward()#retain_graph=True)
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model_params, self._clip_grad_value)
            self._optimizer.step()


            self.model.classif.requires_grad_(True)
            self.model.node_embed.requires_grad_(False)
            self.model.encoder.requires_grad_(False)
            self.model.end_conv.requires_grad_(False)
            self.optimizer_D.zero_grad() # 鉴别器优化器

            pred, dis_out = self.model(X, label)
            label = self._inverse_transform(label)

            # 针对450个采样节点
            mask_value_single = torch.tensor(0)
            if label.min() < 1:
                mask_value_single = label.min()
            if self._iter_cnt == 0:
                print('check single mask value', mask_value_single)

            mape_data = calcu_mape2(pred, label1, mask_value_single)
            yl_label = give_yl_label2(mape_data, yl_values)
            loss2_2 = self.criterion(dis_out, yl_label.float()) # 鉴别器损失
            
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


            '''重新采样部分'''
            # sample_list = optimize_selection(self.district13_road_index, total_nodes, self.sample_num, node_count_global, self.n_number)
            sample_list, node_count_global = optimize_selection(self.district13_road_index, self.sample_num, node_count_global)
            sample_map = sum_map(sample_list, self.sample_num)
            sample_dict = sample_map_district(sample_list, self.district13_road_index)
            district_nodes = get_district_nodes(sample_dict, sample_map)
            
            self._iter_cnt += 1
            batch_count += 1

        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse), np.mean(dis_loss), np.mean(static_fair), np.mean(dynamic_fair) # 一个epoch内！多个batch所以mean
    


    def train(self, sample_list, sample_map, sample_dict, district_nodes, yl_values):
        self._logger.info('Start training!')


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
        node_count_global = calcute_global_dic(node_count_global, sample_list)


        batch_count = 0
        with torch.no_grad():
            for X, label in self._dataloader[mode + '_loader'].get_iterator():
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                X, label= X[:, :, sample_list, :], label[:, :, sample_list, :] # (b,t,938,ci)--->(b,t,450,ci)
                b,t,n,c = X.shape # n是采样的数目,450

                pred, dis_out = self.model(X, label)
                dis_labels = torch.where(dis_out > 0.5, torch.ones_like(dis_out), torch.full_like(dis_out, -1)).reshape(b,t, n, -1) # 使得dis_label具有梯度 .requires_grad_()
            
                pred, label1 = self._inverse_transform([pred, label])


                new_pred = torch.zeros(b, t, len(list(district_nodes.keys())), c)
                new_label = torch.zeros(b, t, len(list(district_nodes.keys())), c)
                for region, nodes_list in district_nodes.items(): # region是str '0' # sample2.py中所有round+0.5确保0-12均有采样，即有0-12
                    new_pred[:, :, int(region)] = pred[:, :, nodes_list].mean(dim=2)#, keepdim=True)#.mean(dim=2)
                    new_label[:, :, int(region)] = label1[:, :, nodes_list].mean(dim=2)#, keepdim=True) #.mean(dim=2)
            
                # 针对13个区域
                mask_value = torch.tensor(0)
                if new_label.min() < 1:
                    mask_value = new_label.min()
                if self._iter_cnt == 0:
                    print('check mask value', mask_value)

                # 针对450个采样节点
                mask_value_single = torch.tensor(0)
                if label1.min() < 1:
                    mask_value_single = label1.min()
                if self._iter_cnt == 0:
                    print('check single mask value', mask_value_single)

                mape_data = calcu_mape2(pred, label1, mask_value_single)
                yl_label = give_yl_label2(mape_data, yl_values)
                loss2 = self.criterion(dis_out, yl_label.float()) # 鉴别器损失

                loss1 = self._loss_fn(new_pred, new_label, mask_value)
                mape = masked_mape(new_pred, new_label, mask_value).item()
                rmse = masked_rmse(new_pred, new_label, mask_value).item()
                loss3, values_region = static_cal(new_pred, new_label) # 静态公平正则化
                loss4 = dynamic_cal5_global(dis_labels, dis_out, sample_map, district_nodes) 
                # loss4, node_count_global = dynamic_cal5_global_xhbl(dis_labels, dis_out, sample_map, district_nodes, self.district13_road_index, node_count_global) # 更新了node_count_global
            

                loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, Loss1: {:.4f}, Dis Loss: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'
                self._logger.info(loss_message.format(epoch + 1, batch_count+1, loss1, loss2, loss3, loss4))

                e_loss.append(loss1.item())
                e_mape.append(mape)
                e_rmse.append(rmse)
                edis_loss.append(loss2.item())
                estatic_fair.append(loss3.item())
                edynamic_fair.append(loss4.item())

                
                sample_list, node_count_global = optimize_selection(self.district13_road_index, self.sample_num, node_count_global)
                # sample_list = optimize_selection(self.district13_road_index, total_nodes, self.sample_num, node_count_global, self.n_number)
                sample_map = sum_map(sample_list, self.sample_num)
                sample_dict = sample_map_district(sample_list, self.district13_road_index)
                district_nodes = get_district_nodes(sample_dict, sample_map)

                self._iter_cnt += 1
                batch_count += 1

        if mode == 'val':
            mvalid_loss, mvalid_mape, mvalid_rmse, mvalidd_loss, mvalid_sfair, mvalid_dfair = \
                np.mean(e_loss), np.mean(e_mape), np.mean(e_rmse), np.mean(edis_loss), np.mean(estatic_fair), np.mean(edynamic_fair)
                
            return mvalid_loss, mvalid_mape, mvalid_rmse, mvalidd_loss, mvalid_sfair, mvalid_dfair

        elif mode == 'test':
    
            log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test D_Loss: {:.4f}, Test SFair: {:.4f}, Test DFair: {:.4f}'
            self._logger.info(log.format(np.mean(e_loss), np.mean(e_rmse), np.mean(e_mape), np.mean(edis_loss), np.mean(estatic_fair), np.mean(edynamic_fair)))
