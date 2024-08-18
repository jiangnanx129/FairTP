import torch
import torch.nn as nn
import torch.nn.functional as F
from src.base.model import BaseModel

class AGCRN(BaseModel):
    '''
    Reference code: https://github.com/LeiBAI/AGCRN
    '''
    def __init__(self, embed_dim, rnn_unit, num_layer, cheb_k, **args):
        super(AGCRN, self).__init__(**args)
        self.node_embed = nn.Parameter(torch.randn(self.node_num, embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(self.input_dim, rnn_unit, cheb_k, embed_dim, num_layer)

        self.end_conv = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(1, rnn_unit), bias=True) # 1对应output[:, -1:, :, :]的-1
        

    def forward(self, source, label=None):  # (b, t, n, f)
        bs, _, node_num, _ = source.shape
        init_state = self.encoder.init_hidden(bs, node_num)
        # print("1:", init_state.shape) # (num_layer, b, N, rnn_unit)
        output, _ = self.encoder(source, init_state, self.node_embed)
        # print("2:", output.shape) # (b, t, N, rnn_unit)
        output = output[:, -1:, :, :] # 
        # print("3:", output.shape) # (b, 1, N, rnn_unit)
        pred = self.end_conv(output)
        # print("4:", pred.shape) # (b, t, N, 1)
        return pred


class AVWDCRNN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, num_layer):
        super(AVWDCRNN, self).__init__()
        assert num_layer >= 1, 'At least one DCRNN layer in the Encoder.'
        self.input_dim = dim_in
        self.num_layer = num_layer
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layer):
            self.dcrnn_cells.append(AGCRNCell(dim_out, dim_out, cheb_k, embed_dim))


    def forward(self, x, init_state, node_embed):
        
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layer):
            state = init_state[i]
            # print("bb:", state.shape, node_embed.shape) # bb: torch.Size([64, 450, 64]) torch.Size([450, 10])
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embed)
                # print("1:", state.shape) # 1: torch.Size([64, 450, 64])
                inner_states.append(state)
            # print("11:", len(inner_states))
            output_hidden.append(state) # 只添加某一层的最后一个state
            # print("2:", len(output_hidden))
            current_inputs = torch.stack(inner_states, dim=1)
            # print("3:", current_inputs.shape)
        return current_inputs, output_hidden


    def init_hidden(self, batch_size, node_num):
        init_states = []
        for i in range(self.num_layer):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size, node_num))
        return torch.stack(init_states, dim=0)


class AGCRNCell(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)


    def forward(self, x, state, node_embed):
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embed))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embed))
        h = r*state + (1-r)*hc
        return h


    def init_hidden_state(self, batch_size, node_num):
        return torch.zeros(batch_size, node_num, self.hidden_dim)


class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))


    def forward(self, x, node_embed):
        node_num = node_embed.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embed, node_embed.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]

        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embed, self.weights_pool)
        bias = torch.matmul(node_embed, self.bias_pool)
        x_g = torch.einsum("knm,bmc->bknc", supports, x)
        x_g = x_g.permute(0, 2, 1, 3)
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias
        return x_gconv