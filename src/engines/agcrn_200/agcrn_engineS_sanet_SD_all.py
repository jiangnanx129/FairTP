import torch
from src.base.engine import BaseEngine
from src.utils.metrics import masked_mape, masked_rmse, masked_mpe, masked_mae, masked_mae_region, masked_mape_region, cal_RSF
# from src.utils.metrics_region import masked_mae2, masked_mpe2, masked_mpe3, masked_mape33
import time
import numpy as np
# from src.engines.IDF_baselines import *
from src.engines.sample_T_single import *
from src.engines.sample_optimal_greedy_T import *  # 先区域后节点
import pickle




class AGCRN_Engine(BaseEngine):
    def __init__(self, T_dynamic, sample_num, district13_road_index, a_loss2, **args):
        super(AGCRN_Engine, self).__init__(**args)
        for p in self.model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)

        self.T_dynamic = T_dynamic
        self.sample_num = sample_num
        self.district13_road_index = district13_road_index
        self.a_loss2 = a_loss2

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = 'final_model_s{}_SD_3_1e-3_sanet_2.pt'.format(self._seed)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))
    
    def load_model(self, save_path):
        filename = 'final_model_s{}_SD_3_1e-3_sanet_2.pt'.format(self._seed)
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename)))


    def train_batch(self, sample_list, time_T, yl_values, sample_map, district_nodes, epoch):
        self.model.train()

        train_loss = []
        train_mape = []
        train_mape2, train_mpe2, train_mae2 = [],[],[]
        train_rmse = []
        static_fair, dynamic_fair = [],[]
        '''新公平正则化'''
        fairst_fair = []

        static_fair_mae = []
        batch_count = 0
        self._dataloader['train_loader'].shuffle()
        for X, label in self._dataloader['train_loader'].get_iterator():
            self._optimizer.zero_grad()
            
            # X (b, t, n, f), label (b, t, n, 1)
            X, label = self._to_device(self._to_tensor([X, label]))
            '''新加的采样模块，不是直接读取采样后的数据，而是采样'''
            b,t,n,c = X.shape # n是采样的数目

            pred = self.model(X, label)
            pred, label = self._inverse_transform([pred, label])
            # 不用反归一化，因为在data/generate_data_for_training.py中归一化后所有数据为0，注释了归一化的代码

            b,t,n,c = X.shape
            new_pred = torch.zeros(b, t, len(list(district_nodes.keys())), c)
            new_label = torch.zeros(b, t, len(list(district_nodes.keys())), c)
            for region, nodes_list in district_nodes.items(): # region是str '0'
                new_pred[:, :, int(region)] = pred[:, :, nodes_list].mean(dim=2)#, keepdim=True)#.mean(dim=2)
                new_label[:, :, int(region)] = label[:, :, nodes_list].mean(dim=2)#, keepdim=True) #.mean(dim=2)

            
            '''节点总数'''
            total_num = sum(len(value) for value in district_nodes.values()) # sum(list(district_nodes.values()), []) # 200
            # print(total_num) # 200
            region_ratio = [len(value) / total_num for value in district_nodes.values()]
            loss2 = sanet_cal(new_pred, new_label, region_ratio)

            
            

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

            loss1 = self._loss_fn(new_pred, new_label, mask_value)
            mape = masked_mape(new_pred, new_label, mask_value).item()
            rmse = masked_rmse(new_pred, new_label, mask_value).item()
            # lpe, loss3_mape, loss3_mae = static_cal(new_pred,new_label) # 静态公平正则化
            
            # loss3_mae = static_cal2_mae(new_pred, new_label, mask_value)
            # loss3_mape = static_cal2_mape(new_pred, new_label, mask_value)
            # print(loss3_mae, loss3_mae.requires_grad, loss3_mape, loss3_mape.requires_grad) # tensor(8.7076, grad_fn=<DivBackward0>) True tensor(0.2452, grad_fn=<DivBackward0>) True

            loss3_mape= static_cal(new_pred,new_label,self._device)
            loss4=-1
            dynamic_fair.append(loss4)

            loss = loss1+ self.a_loss2*loss2
            
            
            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss.item())
            train_mape.append(mape)
            # train_mape2.append(mape2)
            # train_mpe2.append(mpe2)
            # train_mae2.append(mae2)

            train_rmse.append(rmse)
            # static_fair_mae.append(loss3_mae.item()) # mpe
            static_fair.append(loss3_mape.item()) # mae
            fairst_fair.append(loss2.item())

            loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, mape2:{:.4f}, Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}, Loss3_mape: {:.4f}'
            self._logger.info(loss_message.format(epoch + 1, batch_count+1, mape2, loss, loss1, loss2, loss3_mape))


            self._iter_cnt += 1
            batch_count += 1
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_mape2), np.mean(train_mpe2), np.mean(train_mae2), np.mean(train_rmse), np.mean(static_fair), np.mean(fairst_fair) # , np.mean(dynamic_fair)



    def train(self, sample_list, time_T, yl_values, sample_map, district_nodes):
        self._logger.info('Start training!')
        
        yl_list_train, yl_list_train_node = [],[]
        yl_list_val, yl_list_val_node = [],[]
        yl_list_train2, yl_list_val2 = [],[]
        
        wait = 0
        min_loss = np.inf
        for epoch in range(self._max_epochs):
            t1 = time.time()
            mtrain_loss, mtrain_mape, mtrain_mape2, mtrain_mpe2, mtrain_mae2, mtrain_rmse, mtrain_sfair, mtrain_fairst = self.train_batch(sample_list, time_T, yl_values, sample_map, district_nodes, epoch)
            t2 = time.time()

            v1 = time.time()
            mvalid_loss, mvalid_mape, mvalid_mape2, mvalid_mpe2, mvalid_mae2, mvalid_rmse, mvalid_sfair, mvalid_fairst = self.evaluate('val', sample_list, time_T, yl_values, sample_map, district_nodes, epoch)
            v2 = time.time()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            message = 'Epoch: {:03d}, Train mape2: {:.4f}, Train mpe2: {:.4f}, Train mae2: {:.4f}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train SFair_mape: {:.4f}, Fair_fairst: {:.4f}, Valid mape2: {:.4f}, Valid mpe2: {:.4f}, Valid mae2: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Valid SFair: {:.4f}, Fair_fairst: {:.4f}, Train Time: {:.4f}s/epoch, Valid Time: {:.4f}s, LR: {:.4e}'
            self._logger.info(message.format(epoch + 1, mtrain_mape2, mtrain_mpe2, mtrain_mae2, mtrain_loss, mtrain_rmse, mtrain_mape, mtrain_sfair, mtrain_fairst, \
                                             mvalid_mape2, mvalid_mpe2, mvalid_mae2, mvalid_loss, mvalid_rmse, mvalid_mape, mvalid_sfair, mvalid_fairst, \
                                             (t2 - t1), (v2 - v1), cur_lr))

            # yl_list_train.append(mtrain_mpe2.item())
            # yl_list_val.append(mvalid_mpe2.item())
            # yl_list_train_node.append(mtrain_mae2.item())
            # yl_list_val_node.append(mvalid_mae2.item())
            # yl_list_train2.append(mtrain_mape2.item())
            # yl_list_val2.append(mvalid_mape2.item())


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
        # filename1 = 'ylvalue_s{}_HK_3_1e-2_node_mpe2_S_all2.pkl'.format(self._seed) 
        # with open(os.path.join(self._save_path, filename1), 'wb') as f:
        #     pickle.dump((yl_list_train, yl_list_val), f)

        # filename2 = 'ylvalue_s{}_HK_3_1e-2_node_mae2_S_all2.pkl'.format(self._seed)
        # with open(os.path.join(self._save_path, filename2), 'wb') as f:
        #     pickle.dump((yl_list_train_node, yl_list_val_node), f)

        # filename1 = 'ylvalue_s{}_HK_3_1e-2_node_mape2_S_all2.pkl'.format(self._seed)
        # with open(os.path.join(self._save_path, filename1), 'wb') as f:
        #     pickle.dump((yl_list_train2, yl_list_val2), f)


        self.evaluate('test', sample_list, time_T, yl_values, sample_map, district_nodes, epoch)



    def evaluate(self, mode, sample_list, time_T, yl_values, sample_map, district_nodes, epoch):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()

        preds = []
        labels = []
        e_loss = []
        e_mape = []
        e_mape2, e_mpe2, e_mae2 = [],[],[]
        e_rmse = []
        estatic_fair, edynamic_fair =[],[]

        estatic_fairst = []

        estatic_fair_mae = []

        loss_region_list, loss_region_list_mape = [],[]
        
        mpe_region_list = []
        mape_region_list = []

        batch_count = 0
        with torch.no_grad():
            for X, label in self._dataloader[mode + '_loader'].get_iterator():
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))

                b,t,n,c = X.shape # n是采样的数目,450

                pred = self.model(X, label)
                pred, label = self._inverse_transform([pred, label])

                
                b,t,n,c = X.shape
                new_pred = torch.zeros(b, t, len(list(district_nodes.keys())), c)
                new_label = torch.zeros(b, t, len(list(district_nodes.keys())), c)
                for region, nodes_list in district_nodes.items(): # region是str '0' # sample2.py中所有round+0.5确保0-12均有采样，即有0-12
                    new_pred[:, :, int(region)] = pred[:, :, nodes_list].mean(dim=2)#, keepdim=True)#.mean(dim=2)
                    new_label[:, :, int(region)] = label[:, :, nodes_list].mean(dim=2)#, keepdim=True) #.mean(dim=2)

                '''节点总数'''
                total_num = sum(len(value) for value in district_nodes.values()) # sum(list(district_nodes.values()), []) # 200
                # print(total_num) # 200
                region_ratio = [len(value) / total_num for value in district_nodes.values()]
                loss2 = sanet_cal(new_pred, new_label, region_ratio)


                mask_value = torch.tensor(0)
                if new_label.min() < 1:
                    mask_value = new_label.min()

                '''看13个区域的情况'''
                loss_region = masked_mae_region(new_pred, new_label, mask_value)
                loss_region_mape = masked_mape_region(new_pred, new_label, mask_value)


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
                # loss3, loss3_mape, loss3_mae = static_cal(new_pred,new_label) # 静态公平正则化
                # loss3_mae = static_cal2_mae(new_pred, new_label, mask_value)
                # loss3_mape = static_cal2_mape(new_pred, new_label, mask_value)
                # lpe, loss3_mape, loss3_mae = static_cal(new_pred,new_label) # 静态公平正则化
                loss3_mape = static_cal(new_pred,new_label, self._device) # 静态公平正则化
                
                

                loss4=-1
                edynamic_fair.append(loss4)

                e_loss.append(loss.item())
                e_mape.append(mape)
                e_mape2.append(mape2)
                e_mpe2.append(mpe2)
                e_mae2.append(mae2)

                e_rmse.append(rmse)
                # estatic_fair_mae.append(loss3_mae.item())
                estatic_fair.append(loss3_mape.item())
                estatic_fairst.append(loss2.item())

                loss_message = 'Epoch: {:03d}, Batch_num:{:03d}, mape2: {:.4f}, mpe2: {:.4f}, mae2: {:.4f}, Loss1: {:.4f}, Loss3_mape: {:.4f}, Loss2: {:.4f}'
                self._logger.info(loss_message.format(epoch + 1, batch_count+1, mape2, mpe2, mae2, loss, loss3_mape, loss2))

                if mode == "test":
                    loss_region_list.append(loss_region)
                    loss_region_list_mape.append(loss_region_mape)

    
                batch_count += 1

        if mode == 'val':
            mae, mape, mape2, mpe2, mae2, rmse, mvalid_sfair, mvalid_fairst= np.mean(e_loss), np.mean(e_mape), np.mean(e_mape2), np.mean(e_mpe2), np.mean(e_mae2), np.mean(e_rmse), np.mean(estatic_fair), np.mean(estatic_fairst)
            return mae, mape, mape2, mpe2, mae2, rmse, mvalid_sfair, mvalid_fairst

        elif mode == 'test':
    
            log = 'Average Test mape2: {:.4f}, mpe2: {:.4f}, mae2: {:.4f}, MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test SFair_mape: {:.4f}, Fairst: {:.4f}'
            self._logger.info(log.format(np.mean(e_mape2), np.mean(e_mpe2), np.mean(e_mae2), np.mean(e_loss), np.mean(e_rmse), np.mean(e_mape), np.mean(estatic_fair), np.mean(estatic_fairst)))

            
            loss_region_tensor = torch.stack(loss_region_list,dim=0)
            print(loss_region_tensor.shape)
            mean_values = loss_region_tensor.mean(dim=0)
            print(mean_values)
            self._logger.info("mae:")
            self._logger.info(mean_values)

            loss_region_tensor_mape = torch.stack(loss_region_list_mape,dim=0)
            mean_values_mape = loss_region_tensor_mape.mean(dim=0)
            self._logger.info("mape:")
            self._logger.info(mean_values_mape)

            RSF = cal_RSF(mean_values_mape) # mean_values_mape为 list
            self._logger.info("RSF:")
            self._logger.info(RSF)

            # loss_region_tensor_mpe = torch.stack(mpe_region_list,dim=0)
            # print(loss_region_tensor_mpe.shape)
            # mean_values_mpe = loss_region_tensor_mpe.mean(dim=0)
            # print(mean_values_mpe)
            # self._logger.info("mpe:")
            # self._logger.info(mean_values_mpe)


            # loss_region_tensor_mape = torch.stack(mape_region_list,dim=0)
            # print(loss_region_tensor_mape.shape)
            # mean_values_mape = loss_region_tensor_mape.mean(dim=0)
            # print(mean_values_mape)
            # self._logger.info("mape:")
            # self._logger.info(mean_values_mape)