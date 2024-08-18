import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from src.base.model import BaseModel
from src.utils.graph_algo import normalize_adj_mx

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math


# class FairGNN(BaseModel):

#     def __init__(self, device, n_filters, **args):
#         super(FairGNN,self).__init__(**args)
#         self.device = device
#         nhid = n_filters
#         dropout = 0.5 # args.dropout
#         # self.estimator = GCN(nfeat,args.hidden,1,dropout) # fe # 没有估计器，直接给敏感特征
#         self.GNN = GCN_Body(self.input_dim, nhid, dropout)   
#         self.classifier = nn.Linear(nhid,self.output_dim) 
#         self.adv = nn.Linear(nhid,1) # fa

#         # G_params = list(self.GNN.parameters()) + list(self.classifier.parameters()) + list(self.estimator.parameters())
#         # self.optimizer_G = torch.optim.Adam(G_params, lr = args.lr, weight_decay = args.weight_decay)
#         # self.optimizer_A = torch.optim.Adam(self.adv.parameters(), lr = args.lr, weight_decay = args.weight_decay)

#         # self.args = args
#         # self.criterion = nn.BCEWithLogitsLoss()

#         # self.G_loss = 0
#         # self.A_loss = 0

#     def forward(self,g,x):
#         # s = self.estimator(g,x)
#         h = self.GNN(g,x)
#         y = self.classifier(h)
#         ad = self.adv(h)
#         return y,h, ad



# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GCN, self).__init__()
#         self.body = GCN_Body(nfeat,nhid,dropout)
#         self.fc = nn.Linear(nhid,nclass)

#     def forward(self, g, x):
#         x = self.body(g,x)
#         x = self.fc(x)
#         return x

# # def GCN(nn.Module):
# class GCN_Body(nn.Module):
#     def __init__(self, nfeat, nhid, dropout):
#         super(GCN_Body, self).__init__()

#         self.gc1 = GraphConv(nfeat, nhid)
#         self.gc2 = GraphConv(nhid, nhid)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, g, x):
#         x = F.relu(self.gc1(g, x))
#         x = self.dropout(x)
#         x = self.gc2(g, x)
#         # x = self.dropout(x)
#         return x    
    














# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj): # assume input:(b,t,n,c), adj:(n,n)
        # input = input.permute((1, 0))
        # support = torch.mm(input, self.weight)
        # output = torch.spmm(adj.to(device), support)
        support = torch.einsum('btnc, cm -> btnm', input, self.weight)
        output = torch.einsum('nn, btnm -> btnm', adj, support)
        

        if self.bias is not None:
            output = output + self.bias
            #output = output.permute((1, 0))
            return output
        else:
            #output = output.permute((1, 0))
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN_LSTM_layer(nn.Module):
    def __init__(self, in_feature, out_feature, node_num):
        super(GCN_LSTM_layer, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.node_num = node_num

        self.gcn = GraphConvolution(self.in_feature, self.out_feature)
        # nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True, dropout=0.5)
        self.lstm = nn.LSTM(input_size = self.node_num*self.out_feature, hidden_size = self.node_num*self.out_feature, batch_first=True)#, dropout=0.2) # b,t,n*c
        

    def forward(self, x, adj):
        gcn_out = self.gcn(x, adj) # (b,t,n,c)-->(b,t,n,c1)
        gcn_out = gcn_out.reshape(gcn_out.shape[0], gcn_out.shape[1], -1)
        lstm_out, _ = self.lstm(gcn_out)
        lstm_out = lstm_out.reshape(lstm_out.shape[0], lstm_out.shape[1], -1, self.out_feature)

        return lstm_out
    

class GCN_LSTM(BaseModel):

    def __init__(self, device, n_filters, adj, **args):
        super(GCN_LSTM,self).__init__(**args)

        self.device = device
        self.adj = adj
        self.gcn_lstm1 = GCN_LSTM_layer(in_feature=self.input_dim, out_feature=n_filters, node_num=self.node_num)
        self.gcn_lstm2 = GCN_LSTM_layer(in_feature=n_filters, out_feature=int(n_filters/2), node_num=self.node_num)
        self.gcn_lstm3 = GCN_LSTM_layer(in_feature=int(n_filters/2), out_feature=self.output_dim, node_num=self.node_num)
        self.relu = nn.ReLU()


    def forward(self,x, label=None):
        out1 = self.gcn_lstm1(x,self.adj)  # (b,t,n,c1)
        out2 = self.gcn_lstm2(out1,self.adj)  # (b,t,n,)
        out3 = self.gcn_lstm3(out2,self.adj)

        return out3
