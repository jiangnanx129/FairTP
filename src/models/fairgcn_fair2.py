import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from src.base.model import BaseModel
from src.utils.graph_algo import normalize_adj_mx


class FairGNN(BaseModel):

    def __init__(self, device, n_filters, **args):
        super(FairGNN,self).__init__(**args)
        self.device = device
        nhid = n_filters
        dropout = 0.5 # args.dropout
        # self.estimator = GCN(nfeat,args.hidden,1,dropout) # fe # 没有估计器，直接给敏感特征
        self.GNN = GCN_Body(self.input_dim, nhid, dropout)   
        self.classifier = nn.Linear(nhid,self.output_dim) 
        self.adv = nn.Linear(nhid,1) # fa

        # G_params = list(self.GNN.parameters()) + list(self.classifier.parameters()) + list(self.estimator.parameters())
        # self.optimizer_G = torch.optim.Adam(G_params, lr = args.lr, weight_decay = args.weight_decay)
        # self.optimizer_A = torch.optim.Adam(self.adv.parameters(), lr = args.lr, weight_decay = args.weight_decay)

        # self.args = args
        # self.criterion = nn.BCEWithLogitsLoss()

        # self.G_loss = 0
        # self.A_loss = 0

    def forward(self,g,x):
        # s = self.estimator(g,x)
        h = self.GNN(g,x)
        y = self.classifier(h)
        ad = self.adv(h)
        return y,h, ad



class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.body = GCN_Body(nfeat,nhid,dropout)
        self.fc = nn.Linear(nhid,nclass)

    def forward(self, g, x):
        x = self.body(g,x)
        x = self.fc(x)
        return x

# def GCN(nn.Module):
class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_Body, self).__init__()

        self.gc1 = GraphConv(nfeat, nhid)
        self.gc2 = GraphConv(nhid, nhid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x):
        x = F.relu(self.gc1(g, x))
        x = self.dropout(x)
        x = self.gc2(g, x)
        # x = self.dropout(x)
        return x    
