import torch
import torch.nn as nn
import numpy as np
from src.base.model import BaseModel
from src.utils.graph_algo import normalize_adj_mx

'''
如国庆前商讨，目前最新版本，训练鉴别器，测试时不使用target而是直接用鉴别器
'''

class FSAMPLE(BaseModel):
    '''
    Reference code: https://github.com/chnsh/DCRNN_PyTorch
    '''
    def __init__(self, scaler, logger, dis_output_dim, device, adj_mx, n_filters, max_diffusion_step, filter_type, \
                 num_rnn_layers, cl_decay_steps, use_curriculum_learning=True, **args):
        super(FSAMPLE, self).__init__(**args)
        self.device = device
        self.filter_type = filter_type # 新加，为了forward里adj采样后调用！
        self.adj_mx = adj_mx # (938,938)
        self._logger = logger
        self._scaler = scaler
        # self.supports = self._calculate_supports(adj_mx, filter_type) # dcrnn只有一个，我们场景adj在改变

        self.encoder = DCRNNEncoder(device=device,
                                    node_num=self.node_num,
                                    input_dim=self.input_dim,
                                    hid_dim=n_filters,
                                    max_diffusion_step=max_diffusion_step,
                                    filter_type=filter_type,
                                    num_rnn_layers=num_rnn_layers)

        self.decoder = DCGRUDecoder(device=device,
                                    node_num=self.node_num,
                                    input_dim=self.output_dim,
                                    hid_dim=n_filters,
                                    output_dim=self.output_dim,
                                    max_diffusion_step=max_diffusion_step,
                                    filter_type=filter_type,
                                    num_rnn_layers=num_rnn_layers)
        # 鉴别器 MLP_discriminator！
        self.classif = BinaryClassifier(device=device,
                                        input_dim=n_filters, # 隐藏的维度c‘
                                        dis_output_dim=dis_output_dim)
        
        self.use_curriculum_learning = use_curriculum_learning
        self.cl_decay_steps = cl_decay_steps


    def _calculate_supports(self, adj_mx, filter_type):
        supports = normalize_adj_mx(adj_mx, filter_type, 'coo')

        results = []
        for support in supports:
            results.append(self._build_sparse_matrix(support).to(self.device)) # adj放到gpu
        return results


    def _build_sparse_matrix(self, L):
        shape = L.shape
        i = torch.LongTensor(np.vstack((L.row, L.col)).astype(int))
        v = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))


    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))


    def _inverse_transform(self, tensors): # 执行逆标准化
        def inv(tensor):
            return self._scaler.inverse_transform(tensor)

        if isinstance(tensors, list):
            return [inv(tensor) for tensor in tensors]
        else:
            return inv(tensors)
        

    # 模型训练, 输入X, label，sample_list
    def forward(self, source, target, sample_list, yl_values, iter=None):  # (b, t, n, f)
        
        b, t, n, _ = source.shape
        target_for_yl = target.clone() # 注意！！！
        # print("1. target_for_yl:", target_for_yl.shape) # (b,t,n,c)
        go_symbol = torch.zeros(
            1, b, self.node_num, self.output_dim).to(self.device)

        source = torch.transpose(source, dim0=0, dim1=1)

        target = torch.transpose(
            target[..., :self.output_dim], dim0=0, dim1=1)
        # print("1. target:", target.shape) # (t,b,n,c)
        target = torch.cat([go_symbol, target], dim=0)
        # print("2. target:", target.shape) # (t+1,b,n,c)

        # initialize the hidden state of the encoder
        init_hidden_state = self.encoder.init_hidden(b).to(self.device)

        # 根据sample_list从(938,938)中采样adj
        adj_mx = adj_sample(self.adj_mx, sample_list)
        self.supports = self._calculate_supports(adj_mx, self.filter_type) # 掉用index中的函数

        # last hidden state of the encoder is the context
        # print("11. source.requires_grad:", source.requires_grad)
        # print("22. target.requires_grad:", target.requires_grad)
        context, _ = self.encoder(source, self.supports, init_hidden_state)

        if self.training and self.use_curriculum_learning:
            teacher_forcing_ratio = self._compute_sampling_threshold(iter)
        else:
            teacher_forcing_ratio = 0
        
        # dcgru4_state_tensor是隐藏状态
        outputs, dcgru4_state_tensor = self.decoder( # outputs是(t+1,b,n*co) dcgru4_state_tensor是(t,b,n*ch)
            target, self.supports, context, teacher_forcing_ratio=teacher_forcing_ratio)
        # print("33. outputs.requires_grad:", outputs.requires_grad)
        # print("44. dcgru4_state_tensor.requires_grad:", dcgru4_state_tensor.requires_grad)
        o = outputs[1:, :, :].permute(1, 0, 2).reshape(b, t, n, self.output_dim) # dcrnn的输出o, (b,t,n,co)

        # 为了和ground_truth比较，在此处进行归一化
        o, target_for_yl = self._inverse_transform([o, target_for_yl])
        mask_value = torch.tensor(0)
        if target_for_yl.min() < 1:
            mask_value = target_for_yl.min()
        
        mape_data = calcu_mape2(o, target_for_yl,mask_value) # 没有反归一化，o为(b,t,n,co)|target_for_yl为(b,t,n,1) 输出(b,t,n,1)
        # print("-------------------",torch.mean(mape_data))
        yl_label = give_yl_label2(mape_data, yl_values) # 返回yl_label为(b,n,co)
        # 改动，一半优 一半牺牲
        # yl_label = give_yl_label_half(mape_data) # (b,t,n,o)
        # print("a. outputs.requires_grad:", dcgru4_state_tensor.requires_grad)
        '''
        1. 先鉴别器，后encoder+decoder，不需要detach()
        2. 只做一次self.model, 要加上detach(): dcgru4_state_tensor.reshape(t,b,n,-1).detach()
        '''
        # a_input = dcgru4_state_tensor.reshape(t,b,n,-1).clone()
        dis_out = self.classif(dcgru4_state_tensor.reshape(t,b,n,-1).clone()) # 返回的x为(t*b*n,co) .clone()  .detach()
        # print("b. outputs.requires_grad:", dcgru4_state_tensor.reshape(t,b,n,-1).detach().requires_grad) # F
        
        dis_labels = torch.where(dis_out > 0.5, torch.ones_like(dis_out), torch.full_like(dis_out, -1)).reshape(b,t, n, -1) # 使得dis_label具有梯度 .requires_grad_()
        # dis_labels = torch.where(dis_out > 0.5, torch.ones_like(dis_out), torch.full_like(dis_out, -1)).reshape(t,b,n,-1).permute(1, 0, 2,3) # 鉴别器输出的标签(t*b*n,co),为-1/1
        # print("b2. dis_out.requires_grad:", dis_out.requires_grad, dis_labels.requires_grad)
        
        return o, dis_labels, yl_label, dis_out, mape_data
        '''
        o为反归一化的预测结果，(b,t,n,cout)
        dis_labels为鉴别器输出结果, (b,t,n,cout), 全1或-1
        yl_label为优劣label，根据o和ground truth的比较得出, (b,t,n,cout), 全0或1表示牺牲/受益
        x为鉴别器输出，(t*b*n,cout), 经过sigmoid为0-1的值
        mape_data是根据o和ground truth的比较得出的mape数据值，全为正数，(b,t,n,cout)
        '''


# 根据sample_list, 更新采样的邻接矩阵
def adj_sample(adj_mx, sample_list):
    # adj采样，一直不变，从到到尾都是那450个节点
    # raw_adj = np.load(adj_path938) # 最上面定义了
    new_adj = adj_mx[sample_list] # 8k*8k, select 716*8k
    new_adj = new_adj[:,sample_list]
    # print(new_adj.shape) # 返回 716*716 的邻接矩阵
    return new_adj

def calcu_mape(preds, labels): # 均为(b,t,n,c)-gpu, 在region_map_test.py中有测试
    loss = torch.abs((preds - labels) / labels) # (b,t,n,c)
    #loss = torch.mean(loss, dim=(3)) # (b,t,n)
    return loss # 绝对值

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

def give_yl_label1(m_data): # (b,t,n) 存储n各节点的mape，tensor, 一半y一半l
    # 创建标签张量
    yl_labels = torch.zeros_like(m_data, dtype=torch.float32) # (b,t,n,1)
    # print(m_data.shape)
    # 对每个时间步长进行循环
    for i in range(m_data.shape[1]):
        # 对每个样本进行循环
        for j in range(m_data.shape[0]):
            # 获取当前时间步长下的 MAPE 值排序索引
            sorted_indices = torch.argsort(m_data[j, i]) # 返回元素排序后的索引
        
        # 确定需要设置为 +1 的节点数量
        positive_count = m_data.shape[2] // 2
    
        # 设置节点标签为 +1
        yl_labels[j, i, sorted_indices[:positive_count]] = 1.0
        # 设置节点标签为 -1
        yl_labels[j, i, sorted_indices[positive_count:]] = -1.0
    
    return yl_labels

def give_yl_label_half(m_data): # (b,t,n) 存储n各节点的mape，tensor, 一半y一半l
    b,t,n,_ = m_data.shape
    m_data = m_data.reshape(b*t,-1) 
    # print("1.", m_data)
    for i in range(m_data.shape[0]):
        sorted_indices = torch.argsort(m_data[i]) # (450,) 从小到大的排序的索引
        positive_count = m_data.shape[1] // 2 
        # print("受益：", positive_count)
        m_data[i, sorted_indices[:positive_count]] = 1
        m_data[i, sorted_indices[positive_count:]] = 0

    # print("2.", m_data)
    yl_label = m_data.reshape(b,t,n,-1)
    
    return yl_label

# 0/1牺牲
def give_yl_label2(mape_data, yl_values): # yl_values阈值，大于表示误差大判断为l，小于表示误差小判断为y
    # mape_data[mape_data > yl_values] = -1
    # mape_data[mape_data <= yl_values] = 1 # mape_data是(b,n)
    yl_label = torch.where(mape_data > yl_values, 0, 1) # 误差大就0！表示牺牲
    return yl_label

def give_yl_label3(mape_data, yl_values): # yl_values阈值，大于表示误差大判断为l，小于表示误差小判断为y
    # mape_data[mape_data > yl_values] = -1
    # mape_data[mape_data <= yl_values] = 1 # mape_data是(b,n)
    yl_label = torch.where(mape_data > yl_values, 1, 0) # 误差大就0！表示牺牲
    return yl_label

class BinaryClassifier2(nn.Module):
    def __init__(self, device, input_dim, dis_output_dim):
        super(BinaryClassifier2, self).__init__()
        self.device = device
        self.dis_output_dim = dis_output_dim

        # 定义模型的层
        self.fc = nn.Linear(input_dim, dis_output_dim)  # 全连接层
        self.softmax = nn.Softmax()
        # self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数,将值映射到0-1

    def forward(self, x): # x为(t,b,n,ch)
        t, b, n, ch = x.shape
        # x2 = torch.zeros(t, b, n, self.dis_output_dim).to(self.device)
        # # print("x2:", x2.shape)
        # for i in range (x.shape[0]):
        #     x2[i] = self.fc(x[i].reshape(-1,ch)).reshape(b,n,-1) # (b*n,ch)-->(b,n*2)

        x = self.fc(x.reshape(-1,ch)) # 返回(t*b*n, co)
        x = self.softmax(x)
        return x # 1.shape(t*b*n,co), 2.shape(t,b,n,2)

class BinaryClassifier(nn.Module):
    def __init__(self, device, input_dim, dis_output_dim):
        super(BinaryClassifier, self).__init__()
        self.device = device
        self.dis_output_dim = dis_output_dim

        # 定义模型的层
        self.fc = nn.Linear(input_dim, dis_output_dim)  # 全连接层
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数,将值映射到0-1

    def forward(self, x): # x为(t,b,n,ch)
        t, b, n, ch = x.shape
        # x1 = torch.zeros(t, b, n*self.output_dim).to(self.device)
        # for i in range (x.shape[0]):
        #     x1[i] = self.fc(x[i].reshape(-1,ch)).reshape(b,-1) # (b*n,co)-->(b,n*co)

        # 1. 不走循环. (t,b,n,ch)-->(t*b*n,ch)--过linear>(t*b*n,co)
        x = self.fc(x.reshape(-1,ch)) # 输出(t*b*n,co)
        # # 2. 走循环
        # for i in range (x.shape[0]): # (t,b,n,ch)每次取(b,n,ch)
        #     x[i] = self.fc(x[i].reshape(-1,ch)).reshape(b,n,-1) # (b*n,co)-->(b,n*co)

        x = self.sigmoid(x) # 0-1
        return x # 1.shape(t*b*n,co), 2.shape(t,b,n,co)


class BinaryClassifier3(nn.Module):
    def __init__(self, device, input_dim, dis_output_dim):
        super(BinaryClassifier3, self).__init__()
        self.device = device
        self.dis_output_dim = dis_output_dim

        # 定义模型的层
        self.fc1 = nn.Linear(input_dim*12*64, 64)  # 全连接层
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数,将值映射到0-1

    def forward(self, x): # x为(t,b,n,ch)
        # x_list = []
        # t, b, n, ch = x.shape
        # x = x.reshape(t,b*n, ch)
    
        # for i in range(0,t):
        #     x1 = self.relu(self.fc1(x[i]))
        #     x1 = self.relu(self.fc2(x1))
        #     x_list.append(x1)
        
        # x = torch.stack(x_list, dim=1) # (t,b*n,1)
        # print(x.shape)
        # pirnt()
        
        # x = self.sigmoid(x) # 0-1
        t, b, n, ch = x.shape
        x = x.reshape(n,-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        print(x.shape)
        print()
        return x # 1.shape(t*b*n,co), 2.shape(t,b,n,co)


class MLP_out(nn.Module):
    def __init__(self, device, input_dim, output_dim):
        super(MLP_out, self).__init__()
        self.device = device
        self.output_dim = output_dim
        # 定义模型的层
        self.fc = nn.Linear(input_dim, output_dim)  # 全连接层
       
    def forward(self, x): # x为(t,b,n,ch)
        t, b, n, ch = x.shape
        x1 = torch.zeros(t, b, n, self.output_dim).to(self.device)
        # 1. 不走循环. (t,b,n,ch)-->(t*b*n,ch)--过linear>(t*b*n,co)
        # x = self.fc(x.reshape(-1,ch)) # 输出(t*b*n,co)
        # 2. 走循环
        # print(x[0])
        for i in range (x.shape[0]): # (t,b,n,ch)每次取(b,n,ch)
            x1[i] = self.fc(x[i].reshape(-1,ch)).reshape(b,n,-1) # (b*n,co)-->(b,n*co)
            
        return x1 # 1.shape(t*b*n,co), 2.shape(t,b,n,co)


class DCRNNEncoder(nn.Module):
    def __init__(self, device, node_num, input_dim, hid_dim, \
                 max_diffusion_step, filter_type, num_rnn_layers):
        super(DCRNNEncoder, self).__init__()
        self.num_rnn_layers = num_rnn_layers

        encoding_cells = list()
        encoding_cells.append(DCGRUCell(device=device,
                                        node_num=node_num,
                                        input_dim=input_dim,
                                        num_units=hid_dim,
                                        max_diffusion_step=max_diffusion_step,
                                        filter_type=filter_type))

        for _ in range(1, num_rnn_layers):
            encoding_cells.append(DCGRUCell(device=device,
                                            node_num=node_num,
                                            input_dim=hid_dim,
                                            num_units=hid_dim,
                                            max_diffusion_step=max_diffusion_step,
                                            filter_type=filter_type))

        self.encoding_cells = nn.ModuleList(encoding_cells)


    def forward(self, inputs, supports, initial_hidden_state):
        seq_length = inputs.shape[0]
        batch_size = inputs.shape[1]
        inputs = torch.reshape(
            inputs, (seq_length, batch_size, -1))

        current_inputs = inputs
        output_hidden = []
        for i_layer in range(self.num_rnn_layers):
            hidden_state = initial_hidden_state[i_layer]
            output_inner = []
            for t in range(seq_length):
                _, hidden_state = self.encoding_cells[i_layer](
                    current_inputs[t, ...], supports, hidden_state)
                output_inner.append(hidden_state)
            output_hidden.append(hidden_state)

            current_inputs = torch.stack(output_inner, dim=0)
        return output_hidden, current_inputs


    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_rnn_layers):
            init_states.append(self.encoding_cells[i].init_hidden(batch_size))
        return torch.stack(init_states, dim=0)


class DCGRUDecoder(nn.Module):
    def __init__(self, device, node_num, input_dim, hid_dim, output_dim, \
                 max_diffusion_step, filter_type, num_rnn_layers):
        super(DCGRUDecoder, self).__init__()
        self.device = device
        self.node_num = node_num
        self.output_dim = output_dim
        self.num_rnn_layers = num_rnn_layers

        decoding_cells = list()
        decoding_cells.append(DCGRUCell(device=device,
                                        node_num=node_num,
                                        input_dim=input_dim,
                                        num_units=hid_dim,
                                        max_diffusion_step=max_diffusion_step,
                                        filter_type=filter_type))

        for _ in range(1, num_rnn_layers - 1):
            decoding_cells.append(DCGRUCell(device=device,
                                            node_num=node_num,
                                            input_dim=hid_dim,
                                            num_units=hid_dim,
                                            max_diffusion_step=max_diffusion_step,
                                            filter_type=filter_type))
        
        cell_with_projection = DCGRUCell(device=device,
                                         node_num=node_num,
                                         input_dim=hid_dim,
                                         num_units=hid_dim,
                                         max_diffusion_step=max_diffusion_step,
                                         filter_type=filter_type,
                                         num_proj=output_dim)

        decoding_cells.append(cell_with_projection)
        self.decoding_cells = nn.ModuleList(decoding_cells)


    def forward(self, inputs, supports, initial_hidden_state, teacher_forcing_ratio=0.5):
        seq_length = inputs.shape[0]
        batch_size = inputs.shape[1]
        inputs = torch.reshape(
            inputs, (seq_length, batch_size, -1))

        outputs = torch.zeros(
            seq_length, batch_size, self.node_num*self.output_dim).to(self.device)

        # 专门保存DCGRU4的state的list
        dcgru4_state = []
        current_input = inputs[0]
        for t in range(1, seq_length):
            next_input_hidden_state = []

            for i_layer in range(0, self.num_rnn_layers):
                hidden_state = initial_hidden_state[i_layer]
                output, hidden_state = self.decoding_cells[i_layer](
                    current_input, supports, hidden_state)
                current_input = output
                next_input_hidden_state.append(hidden_state)

            dcgru4_state.append(hidden_state)
            initial_hidden_state = torch.stack(next_input_hidden_state, dim=0)
            outputs[t] = output
            teacher_force = np.random.random() < teacher_forcing_ratio
            current_input = (inputs[t] if teacher_force else output)

        dcgru4_state_tensor = torch.stack(dcgru4_state, dim=0) # (t,b,n,ch)
        return outputs, dcgru4_state_tensor  # dcgru4_state-len=12, 每个元素(b,n,ch) n为节点数目，ch为隐藏维度hid_dim


class DCGRUCell(nn.Module):
    def __init__(self, device, node_num, input_dim, num_units, max_diffusion_step, \
                 filter_type, num_proj=None, activation=torch.tanh, use_gc_for_ru=True):
        super(DCGRUCell, self).__init__()
        self.device = device
        self.node_num = node_num
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._num_proj = num_proj
        self._activation = activation
        self._use_gc_for_ru = use_gc_for_ru

        if filter_type == 'doubletransition':
            supports_len = 2
        else:
            supports_len = 1

        self.dconv_gate = DiffusionGraphConv(node_num=node_num,
                                             supports_len=supports_len,
                                             input_dim=input_dim,
                                             hid_dim=num_units,
                                             output_dim=num_units*2,
                                             max_diffusion_step=max_diffusion_step)

        self.dconv_candidate = DiffusionGraphConv(node_num=node_num,
                                                  supports_len=supports_len,
                                                  input_dim=input_dim,
                                                  hid_dim=num_units,
                                                  output_dim=num_units,
                                                  max_diffusion_step=max_diffusion_step)

        if num_proj is not None:
            self.project = nn.Linear(self._num_units, self._num_proj)


    @property
    def output_size(self):
        output_size = self.node_num * self._num_units
        if self._num_proj is not None:
            output_size = self.node_num * self._num_proj
        return output_size


    def forward(self, inputs, supports, state):
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            fn = self.dconv_gate
        else:
            fn = self._fc
        value = torch.sigmoid(
            fn(inputs, supports, state, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self.node_num, output_size))

        r, u = torch.split(
            value, split_size_or_sections=int(output_size/2), dim=-1)
        r = torch.reshape(r, (-1, self.node_num * self._num_units))
        u = torch.reshape(u, (-1, self.node_num * self._num_units))
        c = self.dconv_candidate(inputs, supports, r * state, self._num_units)

        if self._activation is not None:
            c = self._activation(c)

        output = new_state = u * state + (1 - u) * c

        if self._num_proj is not None:
            batch_size = inputs.shape[0]
            output = torch.reshape(new_state, shape=(-1, self._num_units)) # (b,n*ch)-->(b*n,ch)
            output = torch.reshape(self.project(output), shape=( # project为linear，将(b*n,ch)-->(b*n,co)--reshape>(b,n*co)
                batch_size, self.output_size))
        return output, new_state # output(b,n*co), new_state(b,n*ch)


    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)


    def _fc(self, inputs, state, output_size, bias_start=0.0):
        pass


    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.node_num * self._num_units).to(self.device)
    

class DiffusionGraphConv(nn.Module):
    def __init__(self, node_num, supports_len, input_dim, hid_dim, \
                 output_dim, max_diffusion_step, bias_start=0.0):
        super(DiffusionGraphConv, self).__init__()
        self.node_num = node_num
        self.num_matrices = supports_len * max_diffusion_step + 1
        input_size = input_dim + hid_dim
        self._max_diffusion_step = max_diffusion_step
        self.weight = nn.Parameter(torch.FloatTensor(
            size=(input_size*self.num_matrices, output_dim)))
        self.biases = nn.Parameter(torch.FloatTensor(size=(output_dim,)))

        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.biases.data, val=bias_start)


    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)


    def forward(self, inputs, supports, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self.node_num, -1))
        state = torch.reshape(state, (batch_size, self.node_num, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.shape[2]

        x = inputs_and_state
        x0 = torch.transpose(x, dim0=0, dim1=1)
        x0 = torch.transpose(x0, dim0=1, dim1=2)
        x0 = torch.reshape(
            x0, shape=[self.node_num, input_size * batch_size])
        x = torch.unsqueeze(x0, dim=0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in supports:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)
                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        x = torch.reshape(
            x, shape=[self.num_matrices, self.node_num, input_size, batch_size])
        x = torch.transpose(x, dim0=0, dim1=3)
        x = torch.reshape(
            x, shape=[batch_size * self.node_num, input_size * self.num_matrices])

        x = torch.matmul(x, self.weight)
        x = torch.add(x, self.biases)
        return torch.reshape(x, [batch_size, self.node_num * output_size])