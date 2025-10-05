import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from models import lightformer

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
#         print(x.type(), A.type())
        x = torch.einsum('bwf,vw->bvf',(x, A))
        return x.contiguous()

class nconv2(nn.Module):
    def __init__(self):
        super(nconv2,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=1):
        super(gcn,self).__init__()
        self.nconv = nconv2()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class VNA_Layer(nn.Module): #Virtual node alignment
    def __init__(self, input_dim:int, output_dim:int, dropout=0.3, activation=nn.LeakyReLU(0.2),num_adj=1,cross_client=False,num_nodes=[0,0],kernel='MGC'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_adj=num_adj
        self.nconv=nconv()

        self.dropout=dropout if dropout is not None else None
        
        self.activation = activation if activation is not None else None
        
        self.mlps=nn.ModuleList([nn.Linear(self.input_dim, self.output_dim, bias=True) for i in range(self.num_adj)])
        self.cross_client=cross_client
        self.num_nodes=num_nodes
        self.Transformer_nodes=num_nodes[0]
        self.kernel=kernel
        print('kernel type:',self.kernel)

        if self.cross_client:
            self.Transformer_nodes+=num_nodes[1]
        self.lf=LightGFormer(hid_dim=output_dim,num_node=self.Transformer_nodes) #目前测试双方
        

    def forward(self,A,x):
        res=self.convolution(A,x)       

        if self.activation is not None:
            res = self.activation(res)

        if self.dropout is not None:
            res = F.dropout(res, self.dropout, training=self.training)
        return res
    
    def distance_forward(self,A,x):
        return self.mlps[0](self.nconv(x, A))
    
    def adp_forward(self,A,x):
        return self.mlps[1](self.nconv(x, A))
    
    def att_forward(self,x):
        if not self.cross_client:
            return self.lf(x)
        else:
            extended_nodes=self.Transformer_nodes-x.shape[1]
            shape=x.shape[0],extended_nodes,x.shape[2]
            zero_tensor = torch.zeros(shape, device=x.device)
            extended_x=torch.cat((x,zero_tensor),dim=1)
            result=self.lf(extended_x)
            return result[:,x.shape[1]:]
        
    def convolution(self,A,x):
        if self.kernel in ['Top1Sim','FedSim','SL1','SLK']:
            """APPLY Soft linkage"""
            return self.distance_forward(A[0],x)
        else: #VNA hybrid aggregation
            return self.distance_forward(A[0],x)+self.adp_forward(A[1],x)+self.att_forward(x)

class Local_GCLayer(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, dropout=0.3, activation=nn.LeakyReLU(0.2),num_adj=1,cross_client=False,num_nodes=[0,0],kernel='MGC'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_adj=num_adj
        self.nconv=nconv()

        self.dropout=dropout if dropout is not None else None
        
        self.activation = activation if activation is not None else None
        self.cross_client=False
        self.num_nodes=num_nodes

        self.kernel=kernel
        print('kernel type:',self.kernel)
        #GCN, DiffusionConv,GWN,Transformer.
        if self.kernel in ['GCN']:
            self.mlp=nn.Linear(self.input_dim, self.output_dim, bias=True)
        elif self.kernel in ['GWN']:
            self.gconv=gcn(c_in=self.input_dim,c_out=self.output_dim,dropout=0,support_len=3)
        elif self.kernel in ['Transformer']:
            self.Transformer_nodes=num_nodes[0]
            self.lf=LightGFormer(hid_dim=output_dim,num_node=self.Transformer_nodes)
        else: #DIFFUsion CONV
            self.gconv=gcn(c_in=self.input_dim,c_out=self.output_dim,dropout=0,support_len=2)
        if self.cross_client:
            self.Transformer_nodes+=num_nodes[1]
    def forward(self,A,x):
        res=self.spatial_modeling(A,x)       

        if self.activation is not None:
            res = self.activation(res)

        if self.dropout is not None:
            res = F.dropout(res, self.dropout, training=self.training)
        return res
    
    def gcn_forward(self,A,x):
        return self.mlp(self.nconv(x, A))
    
    def gconv_forward(self,A,x):
        x=x.permute(0,2,1).unsqueeze(-1)
        out=self.gconv(x,A)
        out=torch.squeeze(out).permute(0,2,1)
        return out
    
    def att_forward(self,x):
        return self.lf(x)

    def spatial_modeling(self,A,x):
        """GWN(default), GCN, DiffusionConv, Transformer"""
        #GCN, DiffusionConv,GWN,Transformer.

        if self.kernel in ['GWN']:
            #print('gwn forward')
            return self.gconv_forward([A[0],A[0].T,A[1]],x)

        if self.kernel in ['GCN']:
            #print('GCN')
            return self.gcn_forward(A[0],x)
        
        if self.kernel in ['Trans','att','Transformer']:
            #print('GL-former')
            return self.att_forward(x)
        
        if self.kernel in ['DiffusionConv','Diff']:
            return self.gconv_forward([A[0],A[0].T],x)
        else: 
            'ERROR'
            1/0


class Sim_softlinkageLayer(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, dropout=0.3, activation=nn.LeakyReLU(0.2),num_adj=1,no_sim=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_adj=num_adj
        self.nconv=nconv()
        self.dropout=dropout if dropout is not None else None
        self.activation = activation if activation is not None else None
        
        self.mlp=nn.Linear(self.input_dim, self.output_dim, bias=True)
        self.sim_models=nn.Sequential(nn.Linear(1,16),nn.ReLU(),nn.Linear(16,1))
        self.no_sim=no_sim

    def calculate_sim_adjs(self,A):
        if self.no_sim:
            return A
        old_shape=A.shape
        a=A.reshape(-1,1)
        adj=self.sim_models(a).reshape(old_shape)
        a=normalize_rows(adj)
        return a

    def forward(self,A,x):
        adj=self.calculate_sim_adjs(A)
        res=self.mlp(self.nconv(x, adj))  

        if self.activation is not None:
            res = self.activation(res)

        if self.dropout is not None:
            res = F.dropout(res, self.dropout, training=self.training)
        return res
    

class DPLayer(nn.Module):
    def __init__(self, eps=-1):
        super(DPLayer, self).__init__()
        self.eps = eps
        self.delta=1e-4
        if self.eps==-1 or self.eps==0:
            self.sigma=0
        else:
            self.sigma=np.sqrt(2*np.log(1.25/self.delta))/self.eps

    def forward(self, input):
        norms = torch.norm(input, dim=(-2, -1), p=2)  
        sensitivity = norms / np.sqrt(input.numel()/input.shape[0])
        mask = torch.randn_like(input)*self.sigma*(sensitivity.unsqueeze(-1).unsqueeze(-1))
        return input + mask
    
class GatedKnowledgefusionLayer(nn.Module):   
    def __init__(self, input_dim:int, output_dim:int):
        super().__init__()
        self.SLL=nn.Linear(input_dim,output_dim)
        self.gf=GatedFusion(output_dim)
        
        

    def forward(self,target_emb,source_emb):
        return self.gf(target_emb,source_emb)                        #self.gf(target_emb,source_emb)
    

class GatedFusion(nn.Module):
    def __init__(self, input_size):
        super(GatedFusion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input1, input2):
        transformed1 = self.linear1(input1)
        transformed2 = self.linear2(input2)
        gate_weights = self.sigmoid(transformed1 + transformed2)
        fused_features = gate_weights * input1 + (1 - gate_weights) * input2

        return fused_features
    
class LightGFormer(nn.Module):
    def __init__(self,hid_dim,num_node,num_layer=1):
        super(LightGFormer, self).__init__()

        self.heads = 4
        self.layers = num_layer
        self.hid_dim = hid_dim

        self.attention_layer = lightformer.LightformerLayer(self.hid_dim, self.heads, self.hid_dim * 2)
        self.attention_norm = nn.LayerNorm(self.hid_dim)
        self.attention = lightformer.Lightformer(self.attention_layer, self.layers, self.attention_norm)
        self.lpos = lightformer.LearnedPositionalEncoding(self.hid_dim, max_len=num_node)

    def forward(self, input, mask=None):
        # print('hid_dim: ', self.hid_dim)
        x = input.permute(1, 0, 2)
        #print(x.shape)
        x = self.lpos(x)
        output = self.attention(x, mask)
        output = output.permute(1, 0, 2)
        return output
    

class CausalConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        self._padding = (kernel_size[-1] - 1) * dilation
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=(0, self._padding), dilation=dilation, groups=groups, bias=bias)

    def forward(self, inputs):
        result = super(CausalConv2d, self).forward(inputs)
        if self._padding != 0:
            return result[:, :, :, :-self._padding]
        return result


class DCCLayer(nn.Module):
    """
    dilated causal convolution layer with GLU function
    """
    def __init__(self, c_in,c_out, kernel_size=(1, 2), stride=1, dilation=1):
        super(DCCLayer, self).__init__()

        self.relu = nn.ReLU()
        self.filter_conv = CausalConv2d(c_in, c_out, kernel_size, stride, dilation=dilation)
        self.gate_conv = CausalConv2d(c_in, c_out, kernel_size, stride, dilation=dilation)
        self.bn = nn.BatchNorm2d(c_out, affine=False)

    def forward(self, x,  **kwargs):
        """
        :param x: [batch_size, f_in, N, T]
        :return:
        """
       
        x = self.relu(x)
        filter = torch.tanh(self.filter_conv(x))
        gate = torch.sigmoid(self.gate_conv(x))
        output = filter * gate
        output = self.bn(output)
        # output = F.dropout(output, 0.5, training=self.training)

        return output


######################################################################
# Informer
######################################################################
class InformerLayer(nn.Module):
    def __init__(self, d_model=32, d_ff=32, dropout=0.2, n_heads=4, activation="relu", output_attention=False):
        super(InformerLayer, self).__init__()
        # d_ff = d_ff or 4*d_model
        self.attention = AttentionLayer(
            ProbAttention(False, attention_dropout=dropout, output_attention=output_attention), d_model, n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        # self.pe = PositionalEmbedding(d_model)
        self.d_model = d_model
        #self.set_device=True

    def forward(self, x, attn_mask=None, **kwargs):

        # x = x[0]
        #b, T, N, C = x.shape 
        b, C, N, T = x.shape 
        #print(x.shape)
        x = x.permute(0, 2, 3, 1)  # [64, 207, 12, 32]  B, N, T, C
        x = x.reshape(-1, T, C)  # [64*207, 12, 32]

        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x+y)

        output = output.reshape(b, -1, T, C)
        output = output.permute(0, 3, 1, 2)

        return output



class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=3, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        """
        :param Q: [b, heads, T, d_k]
        :param K: 
        :param sample_k: c*ln(L_k), set c=3 for now
        :param n_top: top_u queries?
        :return: Q_K and Top_k query index
        """
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

        # kernel_size = 3
        # pad = (kernel_size - 1) // 2
        # self.query_projection = SepConv1d(d_model, d_model, kernel_size, padding=pad)
        # self.key_projection = SepConv1d(d_model, d_model, kernel_size, padding=pad)
        # self.value_projection = SepConv1d(d_model, d_model, kernel_size, padding=pad)

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # queries = queries.transpose(-1, 1)
        # keys = keys.transpose(-1, 1)
        # values = values.transpose(-1, 1)
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

def normalize_rows(tensor):
    row_sums = tensor.sum(dim=1)

    normalized_tensor = tensor / row_sums[:, None]

    normalized_tensor = normalized_tensor / normalized_tensor.sum(dim=1)[:, None]

    return normalized_tensor


# if __name__ == "__main__":
#     i1=InformerLayer()
#     i1.to('cuda:0')
#     print(i1.device)