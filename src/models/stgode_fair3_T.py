import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from src.base.model import BaseModel

'''
注意classify有两种形式
'''
class STGODE(BaseModel):
    '''
    Reference code: https://github.com/square-coder/STGODE
    '''
    def __init__(self, **args):   # A_sp, A_se,  
        super(STGODE, self).__init__(**args)
        # spatial graph
        self.sp_blocks = nn.ModuleList()
        for _ in range(3):
            block = nn.ModuleList([
                STGCNBlock(in_channels=self.input_dim, out_channels=[64, 32, 64], node_num=self.node_num),
                STGCNBlock(in_channels=64, out_channels=[64, 32, 64], node_num=self.node_num)
            ])
            self.sp_blocks.append(block)

        # semantic graph
        self.se_blocks = nn.ModuleList()
        for _ in range(3):
            block = nn.ModuleList([
                STGCNBlock(in_channels=self.input_dim, out_channels=[64, 32, 64], node_num=self.node_num),
                STGCNBlock(in_channels=64, out_channels=[64, 32, 64], node_num=self.node_num)
            ])
            self.se_blocks.append(block)

        self.pred = nn.Sequential(
            nn.Linear(self.seq_len * 64, self.horizon * 32),
            nn.ReLU(),
            nn.Linear(self.horizon * 32, self.horizon)
        )

        self.classify = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, sp_adj, se_adj, label=None):  # (b, t, n, f)
        x = x.transpose(1, 2)
        outs = []
        # spatial graph
        for block in self.sp_blocks:
            blk_out = x
            for layer in block:
                blk_out = layer(blk_out, sp_adj)
            outs.append(blk_out)

        # semantic graph
        for block in self.se_blocks:
            blk_out = x
            for layer in block:
                blk_out = layer(blk_out, se_adj)
            outs.append(blk_out)

        outs = torch.stack(outs)
        x = torch.max(outs, dim=0)[0]

        dis_out = self.classify(x)

        x = x.reshape((x.shape[0], x.shape[1], -1))
        x = self.pred(x)
        x = x.unsqueeze(-1).transpose(1, 2)

        return x, dis_out.transpose(1, 2)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, node_num):
        super(STGCNBlock, self).__init__()
        # self.A_hat = A_hat
        self.temporal1 = TemporalConvNet(num_inputs=in_channels,
                                   num_channels=out_channels)
        self.odeg = ODEG(out_channels[-1], 12, time=6)
        self.temporal2 = TemporalConvNet(num_inputs=out_channels[-1],
                                   num_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(node_num)

    '''new_adj'''
    def forward(self, X, new_adj):
        t = self.temporal1(X)
        t = self.odeg(t, new_adj)
        t = self.temporal2(F.relu(t))

        return self.batch_norm(t)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), dilation=(1, dilation_size), padding=(0, padding))
            self.conv.weight.data.normal_(0, 0.01)
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]

        self.network = nn.Sequential(*layers)
        self.downsample = nn.Conv2d(num_inputs, num_channels[-1], (1, 1)) if num_inputs != num_channels[-1] else None
        if self.downsample:
            self.downsample.weight.data.normal_(0, 0.01)


    def forward(self, x):
        y = x.permute(0, 3, 1, 2)
        y = F.relu(self.network(y) + self.downsample(y) if self.downsample else y)
        y = y.permute(0, 2, 3, 1)
        return y


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size


    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class ODEG(nn.Module):
    def __init__(self, feature_dim, temporal_dim, time):
        super(ODEG, self).__init__()
        self.odeblock = ODEblock(ODEFunc(feature_dim, temporal_dim), t=torch.tensor([0, time]))


    def forward(self, x, new_adj):
        self.odeblock.set_x0(x)
        z = self.odeblock(x, new_adj)
        return F.relu(z)


class ODEblock(nn.Module):
    def __init__(self, odefunc, t=torch.tensor([0,1])):
        super(ODEblock, self).__init__()
        self.t = t
        self.odefunc = odefunc # ODEFunc(feature_dim, temporal_dim)


    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()


    def forward(self, x, new_adj):
        t = self.t.type_as(x)
        self.odefunc.new_adj = new_adj
        z = odeint(self.odefunc, x, t, method='euler')[1]
        return z


class ODEFunc(nn.Module):
    def __init__(self, feature_dim, temporal_dim):
        super(ODEFunc, self).__init__()
        # self.adj = adj # 原始的，但是换了新的参数
        self.x0 = None
        self.alpha = nn.Parameter(0.8 * torch.ones(200)) # adj.shape[1]
        self.beta = 0.6
        self.w = nn.Parameter(torch.eye(feature_dim))
        self.d = nn.Parameter(torch.zeros(feature_dim) + 1)
        self.w2 = nn.Parameter(torch.eye(temporal_dim))
        self.d2 = nn.Parameter(torch.zeros(temporal_dim) + 1)


    def forward(self, t, x):
        alpha = torch.sigmoid(self.alpha).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        xa = torch.einsum('ij, kjlm->kilm', self.new_adj, x)

        d = torch.clamp(self.d, min=0, max=1)
        w = torch.mm(self.w * d, torch.t(self.w))
        xw = torch.einsum('ijkl, lm->ijkm', x, w)

        d2 = torch.clamp(self.d2, min=0, max=1)
        w2 = torch.mm(self.w2 * d2, torch.t(self.w2))
        xw2 = torch.einsum('ijkl, km->ijml', x, w2)

        f = alpha / 2 * xa - x + xw - x + xw2 - x + self.x0
        return f