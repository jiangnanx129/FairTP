# 不是基于鉴别器输出，而是ground_truth和pred
def dynamic_cal6(yl_label2, dis_out, sample_map, district_nodes, district13_road_index):
    b,t,n,c = yl_label2.shape # 全是+-1，根据ground_truth
    dis_out = dis_out.reshape(b,t,n,-1)
    dis_out = dis_out - 0.5 
    
    node_yl_dic = {} 
    node_yl_dic_disout = {} # 键为0-938，值为对应的dis_out的和 0.几

    for v, node_list in district_nodes.items(): # 键区域0-12，值list 0-449
        for node in node_list: # (0-449)
            node_938 = sample_map[node]
            if node_938 not in node_yl_dic_disout:
                node_yl_dic[node_938] = 0
                node_yl_dic_disout[node_938] = 0
                
            node_yl_dic[node_938] += torch.sum(yl_label2[:,:, node, :].flatten())
            node_yl_dic_disout[node_938] += torch.sum(dis_out[:,:, node, :].flatten())

    # 计算动态损失：只基于采样节点
    differences_dlabel = []
    node_keys = list(node_yl_dic.keys()) # 这段时间出现过的采样节点，0-938
    num_nodes = len(node_keys) # 可能len为500(>450) # 一个batch内就是450
    # 计算节点两两之间的差值，只考虑了采样出来的节点
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            diff_dout = torch.abs(node_yl_dic[node_keys[i]] - node_yl_dic[node_keys[j]]) # 两个值相减
            differences_dlabel.append(diff_dout)
    differences_sum_dout = torch.mean(torch.tensor(differences_dlabel)) # 基于dis_out, 0.x计算出的动态损失


    # softmax算采样概率，不除以64和12，node_yl_dic原始506/-42，node_yl_dic_disout原始26/-2
    sum_node_list = list(district13_road_index.values()) # sum_node_list所有节点
    flat_node_list = [num for sublist in sum_node_list for num in sublist] # 展开2维list,所有节点938
    new_node_yl_dic_disout = dict(node_yl_dic_disout) # node_yl_dic的修改不会影响new_node_yl_dic，它是独立的
    for node_938 in flat_node_list:
        # 针对0.x的
        if node_938 in new_node_yl_dic_disout: # node_yl_dic中原本就有，该节点在过去采样中出现过
            continue
        else:
            new_node_yl_dic_disout[node_938] = torch.tensor(0.0, requires_grad=True).to(device)
    

    sorted_new_node_yl_dic_disout = dict(sorted(new_node_yl_dic_disout.items(), key=lambda x: x[0]))
    
    _,norm_tensor = normalize_list(list(sorted_new_node_yl_dic_disout.values()))
    values_node = torch.sigmoid(norm_tensor)

    return differences_sum_dout, values_node
    

def calcu_mape2(preds, labels, null_val): # 均为(b,t,n,c)-gpu, 在region_map_test.py中有测试
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
    yl_label2 = torch.where(mape_data > yl_values, torch.full_like(mape_data, -1), torch.ones_like(mape_data))
    return yl_label, yl_label2



mape_data = calcu_mape2(pred, label,mask_value) # 没有反归一化，o为(b,t,n,co)|target_for_yl为(b,t,n,1) 输出(b,t,n,1)
yl_label, yl_label2 = give_yl_label2(mape_data, yl_values) # 返回yl_label为(b,n,co)
        
    

loss2 = self.criterion(dis_out.reshape(b,t,n,-1), yl_label.float()) # 鉴别器损失，450个节点的准确优劣分类
loss3, values_region = static_cal(new_pred, new_label) # 静态公平正则化
loss4, values_node = dynamic_cal6(yl_label2, dis_out, sample_map, district_nodes, self.district13_road_index)
            
loss = loss1 + 0.5*loss2 + 0.01*loss3 + 0.001*loss4
loss.backward()
if self._clip_grad_value != 0:
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
self._optimizer.step()