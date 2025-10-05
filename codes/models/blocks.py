import torch
import torch.nn as nn
import numpy as np
import os

from models import layers

import torch.nn.functional as F


node_embed_length=128
default_dropout=0.5
lm=3

"""Temporal representation learning"""
class base_model(nn.Module): 
    def __init__(self,  in_dim:int, hid_dim:int,temporal_module='GRU',layer_num=1):
        super().__init__()#graph size
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.mlp=nn.Linear(self.in_dim,hid_dim)
        #temporal_module='GDCC'
        self.tm=temporal_module
        if temporal_module=='Informer':
            self.temporal_module=Informer(self.hid_dim,self.hid_dim,layer_num)#NGRU(self.hid_dim,self.hid_dim)
        elif temporal_module=='LSTM':
            self.temporal_module=NLSTM(self.hid_dim,self.hid_dim,layer_num)
        else:
            self.temporal_module=NGRU(self.hid_dim,self.hid_dim,layer_num)
        print('temporal_module',temporal_module,layer_num)
        self.activation=nn.ReLU()
    
    def forward(self,x):
        embs=self.mlp(x)
        embs=self.activation(embs)
        embs=self.temporal_module(embs)
        return embs
    
    def reset_params(self):
        #print('start reset')
        #nn.init.xavier_uniform_(self.mlp.weight)
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        

class NGRU(nn.Module): #GRU for nodes
    def __init__(self,  in_dim:int,hid_dim:int,layer_num=1):
        super().__init__()
        self.in_dim=in_dim
        self.hid_dim=hid_dim
        self.layer_num=layer_num
        self.gru_cell = nn.ModuleList([nn.GRUCell(in_dim, hid_dim, bias=True) for i in range(self.layer_num)])

    def forward(self,x):
        B,T,N,F=x.shape

        h = [torch.zeros(B*N, self.hid_dim).cuda() for i in range(self.layer_num)]
        for t in range(T):
            h_in=x[:,t,:,:].reshape(-1,self.in_dim)
            h_out=[]
            
            for i in range(self.layer_num):
                h_out_i = self.gru_cell[i](h_in,h[i]) 
                h_out.append(h_out_i)

            h = h_out
    
            
        out = h_out[-1].view(B, N, F)
        return out

class NLSTM(nn.Module): # LSTM for nodes
    def __init__(self, in_dim:int, hid_dim:int, layer_num=1):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.layer_num = layer_num
        self.lstm_cell = nn.ModuleList([nn.LSTMCell(in_dim, hid_dim) for i in range(self.layer_num)])

    def forward(self, x):
        B, T, N, F = x.shape

        h = [torch.zeros(B * N, self.hid_dim).cuda() for i in range(self.layer_num)]
        c = [torch.zeros(B * N, self.hid_dim).cuda() for i in range(self.layer_num)]

        for t in range(T):
            h_in = x[:, t, :, :].reshape(-1, self.in_dim)
            h_out = []
            c_out = []

            for i in range(self.layer_num):
                h_out_i, c_out_i = self.lstm_cell[i](h_in, (h[i], c[i]))
                h_out.append(h_out_i)
                c_out.append(c_out_i)

            h = h_out
            c = c_out
    
        out = h_out[-1].view(B, N, F)
        return out

class Informer(nn.Module): #Informer for nodes
    def __init__(self,  in_dim:int,hid_dim:int, layer_num=1):
        super().__init__()
        self.layer_num=layer_num
        self.informers=nn.ModuleList([layers.InformerLayer(hid_dim,hid_dim) for i in range(layer_num)])
    
    def forward(self,x):
        #B,T,N=x.shape[0],x.shape[1],x.shape[2]
        x = x.permute(0, 3, 2, 1) #BTNF -> BFNT
        for i in range(self.layer_num):
            x=self.informers[i](x)
        #print(x.shape)
        out=x.permute(0, 3, 2, 1)
        #print('1',out.shape)
        return out[:,-1] #B,N,F
 
class Align(nn.Module):
    def __init__(self, input_shape,output_dim:int):
        super().__init__()
        self.input_shape=input_shape
        self.oputput_dim=output_dim
        #self.n_vertex=np.sum(n_vertex)
        self.vertex_type=len(input_shape)

        #print(input_shape)
        participant_layers=[]

        for i in range(self.vertex_type):
            participant_layers.append(nn.Linear(self.input_shape[i][0][2],output_dim))
        self.participant_layers=nn.ModuleList(participant_layers)

    def forward(self,x:torch.Tensor):
        B,T,_=x.shape
        inputs=self.cut_input_features(x)
        outputs=[]
        for i in range(self.vertex_type):
            # print(inputs[i].shape)
            # print(B*T,self.input_shape[i][0][1])
            d=inputs[i].reshape(B*T*self.input_shape[i][0][1],-1) #self.input_shape[i][0][1]: N
            
            aligned_d=self.participant_layers[i](d).reshape(B*T,self.input_shape[i][0][1],-1)
            outputs.append(aligned_d)
            
        #outputs=torch.cat(outputs,dim=1)
        #outputs=F.relu(outputs)
        return outputs
    
    def cut_input_features(self,x):
        inputs=[]
        index=0
        for i in range(self.vertex_type):
            input_f=self.input_shape[i][0][1]*self.input_shape[i][0][2] #N * F
            inputs.append(x[:,:,index:index+input_f])
            index+=input_f
        return inputs


def Hpreprocess(data,data_shape):#data: B,T,sum(N*F)
    P=len(data_shape)
    inputs=[]
    index=0
    B,T=data.shape[0],data.shape[1]
    for p in range(P):
        N,F=data_shape[p][0],data_shape[p][1]
        input_f=N*F #N * F
        #print(input_f)
        inputs.append(data[:,:,index:index+input_f].reshape(B,T,N,F))
        index+=input_f
    #print(inputs[0].shape,inputs[1].shape)
    return inputs

class VNABlock(nn.Module): #Virtual node alignment
    def __init__(self,  input_dim:int, hidden_dim:int,in_nodes,out_nodes,layers_num=lm,protection_param=0.3,kernel='VNA',use_protection=False,test_mla=False):
        super().__init__()#graph size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        adj_length=1
        self.use_adp=True

        if self.use_adp:
            adj_hid_dim=30
            adj_length+=1
            self.nodevec1 = nn.Parameter(torch.randn(out_nodes, adj_hid_dim), requires_grad=True)
            self.nodevec2 = nn.Parameter(torch.randn(adj_hid_dim, in_nodes), requires_grad=True)
        self.num_adj=adj_length

        self.test_multilevel_alignment=test_mla
        l=[layers.VNA_Layer(input_dim=hidden_dim,output_dim=hidden_dim,num_adj=self.num_adj,activation=nn.ReLU(),cross_client=True,num_nodes=[in_nodes,out_nodes],kernel=kernel) for i in range(layers_num)]
        self.layers=nn.ModuleList(l)
        self.use_protection=True
        if protection_param==-1:
            self.use_protection=False
        self.protectionLayer=layers.DPLayer(eps=protection_param)

    def get_adj(self,A):
        if self.use_adp:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            adj=[A,adp]
        else:
            adj=[A]
        return adj
    
    def forward(self, A:torch.Tensor, layer_results:list):
        res=[]
        for i in range(len(layer_results)):
            adj=self.get_adj(A)
            layer_index=i
            if self.test_multilevel_alignment:
                layer_index=0
            embs=self.layers[layer_index](adj,layer_results[i])
            if self.use_protection:
                embs=self.protectionLayer(embs)
            res.append(embs)
        return res
    
    def reset_params(self):
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
    

class Source_GCBlock(nn.Module):
    def __init__(self,  input_dim:int, hidden_dim:int,in_nodes,out_nodes,layers_num=lm,kernel='MGC'):
        super().__init__()#graph size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        adj_length=1
        self.use_adp=True

        if self.use_adp:
            adj_hid_dim=30
            adj_length+=1
            self.nodevec1 = nn.Parameter(torch.randn(out_nodes, adj_hid_dim), requires_grad=True)
            self.nodevec2 = nn.Parameter(torch.randn(adj_hid_dim, in_nodes), requires_grad=True)
        self.num_adj=adj_length
        #layers_num=6
        l=[layers.Local_GCLayer(input_dim=hidden_dim,output_dim=hidden_dim,num_adj=self.num_adj,num_nodes=[in_nodes,out_nodes],kernel=kernel) for i in range(layers_num)]
        self.layers=nn.ModuleList(l)
    
    def forward(self, A:torch.Tensor, x:torch.Tensor):
        adj=self.get_adj(A)
        layer_results=[x]     
        for i in range(len(self.layers)-1):
            
            x=x+self.layers[i](adj,x)
            layer_results.append(x)
        return layer_results
    
    def get_adj(self,A):
        if self.use_adp:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            adj=[A,adp]
        else:
            adj=[A]
        return adj
    
    
    def reset_params(self):
        #print('start reset')
        #nn.init.xavier_uniform_(self.mlp.weight)
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()


class Target_GCBlock(nn.Module):
    def __init__(self,  input_dim:int, hidden_dim:int,in_nodes,out_nodes,layers_num=lm,kernel='MGC'):
        super().__init__()#graph size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        adj_length=1
        self.use_adp=True

        if self.use_adp:
            adj_hid_dim=30
            adj_length+=1
            self.nodevec1 = nn.Parameter(torch.randn(out_nodes, adj_hid_dim), requires_grad=True)
            self.nodevec2 = nn.Parameter(torch.randn(adj_hid_dim, in_nodes), requires_grad=True)
        self.num_adj=adj_length

        #layers_num=6
        l=[layers.Local_GCLayer(input_dim=hidden_dim,output_dim=hidden_dim,num_adj=self.num_adj,num_nodes=[in_nodes,out_nodes],kernel=kernel) for i in range(layers_num)]
        self.layers=nn.ModuleList(l)
        l2=[layers.GatedKnowledgefusionLayer(hidden_dim,hidden_dim) for i in range(layers_num)]
        self.fusion_layers=nn.ModuleList(l2)
        
    
    def forward(self, A:torch.Tensor, x:torch.Tensor):
        adj=self.get_adj(A)
        layer_results=[x]     
        for i in range(len(self.layers)):  
            x=x+self.layers[i](adj,x)
            layer_results.append(x)
        return layer_results
    
    def get_adj(self,A):
        if self.use_adp:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            adj=[A,adp]
        else:
            adj=[A]
        return adj
    
    def forward_with_fusion(self, A:torch.Tensor, x:torch.Tensor,source_embeds:list):
        adj=self.get_adj(A)
        for i in range(len(self.layers)):            
            x=x+self.fusion(self.layers[i](adj,x),source_embeds[i],i)        #,source_embeds[i]
        return x#layer_results[-1]

    
    def fusion(self,target_emb,source_emb,layer_id):
        res=self.fusion_layers[layer_id](target_emb,source_emb)

        return res


#Spatial representation learning & Cross Client VNA block.
class SpatialBlock(nn.Module):
    def __init__(self,  input_dim:int, hidden_dim:int,n_vertexs,protection_param:int,kernel:list,message_passing='all',layer_num=-1,test_mla=False):
        super().__init__()#graph size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        target_nodes,source_nodes=n_vertexs[0],n_vertexs[1]
        #layer_nums=5
        self.target_GC=Target_GCBlock(input_dim,hidden_dim,in_nodes=target_nodes,out_nodes=target_nodes,kernel=kernel[0],layers_num=layer_num)
        self.source_GC=Source_GCBlock(input_dim,hidden_dim,in_nodes=source_nodes,out_nodes=source_nodes,kernel=kernel[0],layers_num=layer_num)

        self.message_passing=message_passing
        self.CCVNA=VNABlock(input_dim,hidden_dim,in_nodes=source_nodes,out_nodes=target_nodes,protection_param=protection_param,kernel=kernel[1],layers_num=layer_num,test_mla=test_mla)

    

    def forward(self,A,x):
        source_embs=self.source_GC(A[2],x[1])  #[L*ND]
        source_embs=self.CCVNA(A[1],source_embs)  #[L*ND]
        source_embs=self.pass_emb(source_embs=source_embs)
        
        x=self.target_GC.forward_with_fusion(A[0],x[0],source_embs)
        return x
    
    def FDML_forward(self,A,x):
        source_embs=self.source_GC(A[2],x[1])  #[L*ND]
        source_embs=self.CCVNA(A[1],source_embs)  #[L*ND]
        source_embs=self.pass_emb(source_embs=source_embs)
        x=self.target_GC.forward(A[0],x[0])[-1]
        return x,source_embs[-1]
    
    def source_forward(self,A,x,method='normal'):
        source_embs=self.source_GC(A[2],x)  #[L*ND]
        #print(source_embs[0].shape)


        source_embs=self.CCVNA(A[1],source_embs)  #[L*ND]

        return source_embs
    
    def reset_passive(self):
        self.source_GC.reset_params()
        self.CCVNA.reset_params()
        return
    
    def passive_parameters(self):
        params=[]
        params+=self.source_GC.parameters()
        params+=self.CCVNA.parameters()
        return params
    
    def pass_emb(self,source_embs):
        if self.message_passing=='all':
            return source_embs
        
        message=[]
        for i in range(len(source_embs)):
            message.append(torch.zeros_like(source_embs[i]))
        
        if self.message_passing=='first':
            message[0]=source_embs[0]
        else:
            message[-1]=source_embs[-1]


        return message
    
class Sim_SLBlock(nn.Module): 
    def __init__(self,  input_dim:int, hidden_dim:int,sim_matrix=None,no_sim=False):
        super().__init__()#graph size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sim_matrix=sim_matrix

        adj_length=1
        self.num_adj=adj_length
        self.no_sim=True

        self.l=layers.Sim_softlinkageLayer(input_dim=hidden_dim,output_dim=hidden_dim,num_adj=self.num_adj,activation=nn.ReLU(),no_sim=no_sim)

    def get_sim(self,A):
        if self.no_sim:
            return A
        return self.sim_matrix.to(A.device) * A
    
    
    def forward(self, A:torch.Tensor, x:torch.Tensor):
        adj=self.get_sim(A)
        res=self.l(adj,x)
        return res
    
    def set_sim_matrix(self,sim_matrix):
        self.sim_matrix=sim_matrix
    

class FedSimBlock(nn.Module):
    def __init__(self,  input_dim:int, hidden_dim:int,sim_matrix=None,no_sim=False):
        super().__init__()#graph size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hetero_conv=Sim_SLBlock(input_dim,hidden_dim,sim_matrix=sim_matrix,no_sim=no_sim)


    def forward(self,A,x):
        sim_result=self.hetero_conv(A[1],x[1])  #[L*ND]
        res=x[0]+sim_result
        return res
    
    def set_sim_matrix(self,sim_matrix):
        self.hetero_conv.set_sim_matrix(sim_matrix)
                                   

                


