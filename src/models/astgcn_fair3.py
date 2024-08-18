import torch
import torch.nn as nn
import torch.nn.functional as F
from src.base.model import BaseModel

class ASTGCN(BaseModel):
    '''
    Reference code: https://github.com/guoshnBJTU/ASTGCN-r-pytorch
    '''
    def __init__(self, device, cheb_poly, order, nb_block, nb_chev_filter, nb_time_filter, time_stride, **args):
        super(ASTGCN, self).__init__(**args)

        self.BlockList = nn.ModuleList([ASTGCN_block(device, self.input_dim, order, nb_chev_filter, nb_time_filter, time_stride, cheb_poly, self.node_num, self.seq_len)])
        self.BlockList.extend([ASTGCN_block(device, nb_time_filter, order, nb_chev_filter, nb_time_filter, 1, cheb_poly, self.node_num, self.seq_len // time_stride) for _ in range(nb_block - 1)])

        self.final_conv = nn.Conv2d(int(self.seq_len / time_stride), self.horizon, kernel_size=(1, nb_time_filter)) # 得是1，不然减少N数目节点数目
        self.classif = BinaryClassifier(device=device, input_dim=nb_time_filter, dis_output_dim = int(nb_time_filter/nb_time_filter), seq_len=self.seq_len, horizon=self.horizon)

    def forward(self, x, label=None):  # (b, t, n, f)
        x = x.permute(0, 2, 3, 1)  # (b, n, f, t)， f == c_in
        # print("1. x:", x.shape) # (b, n, c_in, t)

        for block in self.BlockList:
            x = block(x)
        # print("2. x:", x.shape) # (b, n, c_h, t), c_h == nb_time_filter

        # print("3. x:", x.permute(0, 3, 1, 2).shape) # (b, t, n, c_h), 此处t = int(self.seq_len / time_stride)
        
        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1]  # (b, t, n)

        # output1 = self.final_conv(x.permute(0, 3, 1, 2))
        # print("4. x:", output1.shape) # (b, self.horizon, n, 1) # 此处=1是因为，kernel_size的第二个维度==c_h
        # print("5. x:", output.shape) # (b, self.horizon, n) # 第3个维度==n，是因为kernel_size的第一个维度==1
        dis_out = self.classif(x.permute(0, 3, 1, 2))[:,:,:,-1]

        return output.unsqueeze(-1), dis_out.unsqueeze(-1) # 第2个维度x.permute(0, 3, 1, 2), (b,t,n,c_h), c_h == nb_time_filter
        '''
        1. 模型输出: (b,t,n,c_out)
        2. 隐藏状态：(b,t,n,c_hidd)
        3. 鉴别器输出：(b,t,n,c_out), 经过了softmax
        目前做法：输出btnc直接做sigmoid
        可选做法：输出btnc，区btn在取unsqueeze
        '''

'''参考fsample3.py'''
class BinaryClassifier(nn.Module):
    def __init__(self, device, input_dim, dis_output_dim, seq_len, horizon):
        super(BinaryClassifier, self).__init__()
        self.device = device
        self.dis_output_dim = dis_output_dim
        self.seq_len = seq_len
        self.horizon = horizon
        # 定义模型的层
        '''linear或者Conv2d'''
        # self.fc = nn.Linear(input_dim, dis_output_dim)  # 全连接层
        self.final_conv = nn.Conv2d(self.seq_len, self.horizon, kernel_size=(1, input_dim)) 
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数,将值映射到0-1

    def forward(self, x): # x为(b,t,n,ch)
        # b,t,n,ch = x.shape
        x = self.final_conv(x) # 输出(b,t,n,1)

        x = self.sigmoid(x) # 0-1
        return x # (b,t,n,1)

class ASTGCN_block(nn.Module):
    def __init__(self, device, in_channels, order, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, node_num, seq_len):
        super(ASTGCN_block, self).__init__()
        self.TAt = Temporal_Attention_layer(device, in_channels, node_num, seq_len)
        self.SAt = Spatial_Attention_layer(device, in_channels, node_num, seq_len)
        self.cheb_conv_SAt = cheb_conv_withSAt(order, cheb_polynomials, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)


    def forward(self, x):
        bs, node_num, feature_num, seq_len = x.shape # b,n,c,t

        temporal_At = self.TAt(x)
        x_TAt = torch.matmul(x.reshape(bs, -1, seq_len), temporal_At).reshape(bs, node_num, feature_num, seq_len)

        spatial_At = self.SAt(x_TAt)
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))

        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))
        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        return x_residual


class Temporal_Attention_layer(nn.Module):
    def __init__(self, device, in_channels, node_num, seq_len):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(node_num).to(device))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, node_num).to(device))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(device))
        self.be = nn.Parameter(torch.FloatTensor(1, seq_len, seq_len).to(device))
        self.Ve = nn.Parameter(torch.FloatTensor(seq_len, seq_len).to(device))


    def forward(self, x):
        _, node_num, feature_num, seq_len = x.shape
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        rhs = torch.matmul(self.U3, x)
        product = torch.matmul(lhs, rhs)

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))
        E_normalized = F.softmax(E, dim=1)
        return E_normalized


class Spatial_Attention_layer(nn.Module):
    def __init__(self, device, in_channels, node_num, seq_len):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(seq_len).to(device))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, seq_len).to(device))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(device))
        self.bs = nn.Parameter(torch.FloatTensor(1, node_num, node_num).to(device))
        self.Vs = nn.Parameter(torch.FloatTensor(node_num, node_num).to(device))


    def forward(self, x):
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)
        product = torch.matmul(lhs, rhs)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))
        S_normalized = F.softmax(S, dim=1)
        return S_normalized


class cheb_conv_withSAt(nn.Module):
    def __init__(self, order, cheb_polynomials, in_channels, out_channels):
        super(cheb_conv_withSAt, self).__init__()
        self.order = order
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.device)) for _ in range(order)])


    def forward(self, x, spatial_attention):
        bs, node_num, in_channels, seq_len = x.shape
        outputs = []
        for time_step in range(seq_len):
            graph_signal = x[:, :, :, time_step]
            
            output = torch.zeros(bs, node_num, self.out_channels).to(self.device)
            for k in range(self.order):
                T_k = self.cheb_polynomials[k]
                T_k_with_at = T_k.mul(spatial_attention)
                theta_k = self.Theta[k]
                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)
                output = output + rhs.matmul(theta_k)
            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))