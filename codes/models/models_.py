import os
import torch
import torch.nn as nn
import numpy as np

from models  import blocks

import time


default_tm_hidden_dim=256

class Hnet(nn.Module): 
    def __init__(self,in_dim=1,hid_dim=16,out_dim=-1,n_pred=1,input_shape=[],protection_param=0.0,kernel=['GWN','VNA'],message_passing='all',temporal_module='GRU',layer_num=[1,3],test_mla=False):
        super().__init__()
        self.in_dim=in_dim
        self.hid_dim=hid_dim
        self.out_dim=out_dim[0]
        
        self.n_pred=n_pred
        self.input_shape=input_shape
        self.n_vertex=[input_shape[i][0] for i in range(len(input_shape))]
        self.layer_num=layer_num

        self.target_base_model=blocks.base_model(self.in_dim[0],hid_dim,temporal_module,layer_num[0])
        self.source_base_model=blocks.base_model(self.in_dim[1],hid_dim,temporal_module,layer_num[0])
        self.message_passing=message_passing
        self.ts_mode='None'
        self.tl_l=None
        self.active_tlength=6
        self.passive_tlength=6

        self.gconv=blocks.SpatialBlock(hid_dim,hid_dim,self.n_vertex,protection_param,kernel=kernel,message_passing=self.message_passing,layer_num=layer_num[-1],test_mla=test_mla)
        self.out_mlp=nn.Sequential(nn.Linear(hid_dim,hid_dim),nn.ReLU(),nn.Linear(hid_dim,self.out_dim*n_pred))

    def forward(self,A,x):
        B,T,_=x.shape
        #print(x.shape)
        x=blocks.Hpreprocess(x,self.input_shape)
        #print(x[0].shape,x[1].shape)
        
        #print('Before',x[0].shape,x[1].shape)
        if not self.ts_mode=='None':
            if self.ts_mode=='tl' or self.ts_mode=='tsc':
                x[0]=self.process_timeseries(x[0],is_active=True)
                x[1]=self.process_timeseries(x[1])
            elif self.ts_mode=='tsl':
                x[0]=self.process_timeseries(x[0],is_active=True,compress_mode='mean') #mean to calculate the average parking avalibility
                x[1]=self.process_timeseries(x[1])
            else:
                return


        #print('After',x[0].shape,x[1].shape)
        target_embed=self.target_base_model(x[0])
        source_embed=self.source_base_model(x[1])
        adj=[A[0],A[1],A[2]]
        x=self.gconv(adj,[target_embed,source_embed])
        x=self.out_mlp(x) 
        x=x.reshape(B,-1,self.n_pred,self.out_dim).transpose(1,2)
        return x 

    def process_timeseries(self,data,is_active=False,compress_mode='sum'):
        length=self.tl_l
        #print('process now!')
        if is_active:
            if self.ts_mode=='tl':
                return data[:,-12:] #target timeseries is fixed to length 12
            else: #sampling rate experiment
                #print('active s')
                data=compress_ts_tensor(data, time_length=self.active_tlength, operator=compress_mode)
                return data 
        else:
            if self.ts_mode=='tl':
                return data[:,-length:] #source timeseries is set to length n_his
            else: #sampling rate experiment
                #print('passive s')
                data=compress_ts_tensor(data, time_length=self.passive_tlength, operator=compress_mode)
                return data
        
        
        return data[:,-12:]
    
    def reset_passive(self):
        """reset the passive parties' model"""
        self.source_base_model.reset_params()
        self.gconv.reset_passive()
        return
    
    def reset_temporal(self):
        """reset the passive parties' model"""
        self.source_base_model.reset_params()
        return
    
    def GetSourceOutput(self,A,x,need_preprocess=True):
        # B,T,_=x.shape
        if need_preprocess:
            x=blocks.Hpreprocess(x,self.input_shape)
            source_data=x[1].clone()
            tmp=x[1]
        else:
            source_data=x.clone()
            tmp=x
            
        source_embed=self.source_base_model(tmp)
        adj=[A[0],A[1],A[2]]
        source_output=self.gconv.source_forward(adj,source_embed)
        source_output=torch.stack(source_output)
        return source_data,source_output

    def GetTemporalOutput(self,A,x,need_preprocess=True):
        # B,T,_=x.shape
        if need_preprocess:
            x=blocks.Hpreprocess(x,self.input_shape)
            source_data=x[1].clone()
            tmp=x[1]
        else:
            source_data=x.clone()
            tmp=x
            
        source_output=self.source_base_model(tmp)

        return source_data,source_output
    
    def passive_parameters(self):
        params=[]
        params+=self.source_base_model.parameters()
        params+=self.gconv.passive_parameters()
        return params

    def temporal_parameters(self):
        params=[]
        params+=self.source_base_model.parameters()
        #params+=self.gconv.passive_parameters()
        return params


class Snet(nn.Module): 
    def __init__(self,in_dim=1,hid_dim=16,out_dim=-1,n_pred=1,input_shape=[],kernel=['GWN','VNA'],temporal_module='GRU',layer_num=[1,3]):
        super().__init__()
        self.in_dim=in_dim
        self.hid_dim=hid_dim
        self.out_dim=out_dim[0]
        self.n_pred=n_pred
        self.input_shape=input_shape
        self.n_vertex=[input_shape[i][0] for i in range(len(input_shape))]
        self.layer_num=layer_num

        self.target_base_model=blocks.base_model(self.in_dim[0],hid_dim,temporal_module,layer_num[0])

        self.gconv=blocks.Target_GCBlock(hid_dim,hid_dim,self.n_vertex[0],self.n_vertex[0],kernel=kernel[0],layers_num=layer_num[1])
        self.out_mlp=nn.Sequential(nn.Linear(hid_dim,hid_dim),nn.ReLU(),nn.Linear(hid_dim,self.out_dim*n_pred))

    
    def forward(self,A,x,index=0):
        B,T,_=x.shape
        x=blocks.Hpreprocess(x,self.input_shape)
        # print(x[0].shape,x[1].shape)
        # 1/0
        #print(x[index].shape)
        target_embed=self.target_base_model(x[index])
        adj=A[0]
        # print(adj.shape,target_embed.shape)
        # print(type(adj),type(target_embed))
        x=self.gconv(adj,target_embed)
        x=self.out_mlp(x[-1]) #B,N,P * F
        x=x.reshape(B,-1,self.n_pred,self.out_dim).transpose(1,2)
        return x

class SLFDML(nn.Module):
    def __init__(self,in_dim=1,hid_dim=16,out_dim=-1,n_pred=1,input_shape=[],protection_param=0.0,kernel=['GWN','VNA'],message_passing='all',temporal_module='GRU',layer_num=[1,3],test_mla=False):
        super().__init__()
        self.in_dim=in_dim
        self.hid_dim=hid_dim
        self.out_dim=out_dim[0]
        self.n_pred=n_pred
        self.input_shape=input_shape
        self.n_vertex=[input_shape[i][0] for i in range(len(input_shape))]
        self.layer_num=layer_num

        self.target_base_model=blocks.base_model(self.in_dim[0],hid_dim,temporal_module,layer_num[0]) 
        self.source_base_model=blocks.base_model(self.in_dim[1],hid_dim,temporal_module,layer_num[0])
        self.message_passing=message_passing
        
        FDML_kernel=[kernel[0],'SLK']

        self.gconv=blocks.SpatialBlock(hid_dim,hid_dim,self.n_vertex,protection_param,kernel= FDML_kernel,message_passing='last',layer_num=layer_num[-1])
        self.out_mlp=nn.Sequential(nn.Linear(hid_dim,hid_dim),nn.ReLU(),nn.Linear(hid_dim,self.out_dim*n_pred))
        self.out_mlp2=nn.Sequential(nn.Linear(hid_dim,hid_dim),nn.ReLU(),nn.Linear(hid_dim,self.out_dim*n_pred))

    def forward(self,A,x):
        B,T,_=x.shape
        x=blocks.Hpreprocess(x,self.input_shape)

        target_embed=self.target_base_model(x[0])
        source_embed=self.source_base_model(x[1])

        adj=[A[0],A[1],A[2]]
        x,hetero_x=self.gconv.FDML_forward(adj,[target_embed,source_embed])


        x=self.out_mlp(x)+self.out_mlp2(hetero_x) #B,N,P * F
        x=x.reshape(B,-1,self.n_pred,self.out_dim).transpose(1,2)
        return x
    
    def reset_passive(self):
        """reset the passive parties' model"""
        self.source_base_model.reset_params()
        self.gconv.reset_passive()
        return
    

    def reset_temporal(self):
        """reset the passive parties' model"""
        self.source_base_model.reset_params()
        return
    
    def GetSourceOutput(self,A,x,need_preprocess=True):
        """输入的结果是需要攻击的"""
        # B,T,_=x.shape
        if need_preprocess:
            x=blocks.Hpreprocess(x,self.input_shape)
            source_data=x[1].clone()
            tmp=x[1]
        else:
            source_data=x.clone()
            tmp=x
        # print(x[0].shape,x[1].shape)
        # 1/0
            
        source_embed=self.source_base_model(tmp)
        adj=[A[0],A[1],A[2]]
        source_output=self.gconv.source_forward(adj,source_embed)
        source_output=torch.stack(source_output)

        return source_data,source_output
    

    def GetTemporalOutput(self,A,x,need_preprocess=True):
        """输入的结果是需要攻击的"""
        # B,T,_=x.shape
        if need_preprocess:
            x=blocks.Hpreprocess(x,self.input_shape)
            source_data=x[1].clone()
            tmp=x[1]
        else:
            source_data=x.clone()
            tmp=x
        # print(x[0].shape,x[1].shape)
        # 1/0
            
        source_output=self.source_base_model(tmp)
        #print(type(source_output))
        #print(source_output.shape)
    
        #source_output=torch.stack(source_output)

        return source_data,source_output
    
    def passive_parameters(self):
        params=[]
        params+=self.source_base_model.parameters()
        params+=self.gconv.passive_parameters()
        return params

    def temporal_parameters(self):
        params=[]
        params+=self.source_base_model.parameters()
        #params+=self.gconv.passive_parameters()
        return params

class FedSim(nn.Module): 
    def __init__(self,in_dim=1,hid_dim=16,out_dim=-1,n_pred=1,input_shape=[],protection_param=0.0,kernel=['GWN','VNA'],message_passing='all',temporal_module='GRU',layer_num=[1,3],test_mla=False,arg=None):
        super().__init__()
        self.in_dim=in_dim
        self.hid_dim=hid_dim
        self.out_dim=out_dim[0]
        self.n_pred=n_pred
        self.input_shape=input_shape
        self.n_vertex=[input_shape[i][0] for i in range(len(input_shape))]
        self.layer_num=layer_num
        self.target_base_model=blocks.base_model(self.in_dim[0],hid_dim,temporal_module,layer_num[0])
        self.source_base_model=blocks.base_model(self.in_dim[1],hid_dim,temporal_module,layer_num[0])
        self.no_sim=arg.no_sim
        self.gconv=blocks.FedSimBlock(hid_dim,hid_dim,sim_matrix=None,no_sim=self.no_sim)
        self.out_mlp=nn.Sequential(nn.Linear(hid_dim,hid_dim),nn.ReLU(),nn.Linear(hid_dim,hid_dim),nn.ReLU(),nn.Linear(hid_dim,self.out_dim*n_pred)) 
        """MLP for output, There is no spatial layers in FedSim because FedSim do not support intra-client data correlation modeling."""
    
    def set_sim_matrix(self,m):
        self.gconv.set_sim_matrix(m)

    def forward(self,A,x):
        B,T,_=x.shape
        x=blocks.Hpreprocess(x,self.input_shape)
        target_embed=self.target_base_model(x[0])
        source_embed=self.source_base_model(x[1])
        adj=[A[0],A[1],A[2]]
        x=self.gconv(adj,[target_embed,source_embed])
        x=self.out_mlp(x)
        x=x.reshape(B,-1,self.n_pred,self.out_dim).transpose(1,2)
        return x 
    
    def reset_passive(self):
        self.source_base_model.reset_params()
        self.gconv.reset_passive()
        return
    
    def reset_temporal(self):
        self.source_base_model.reset_params()
        return
    
    def GetSourceOutput(self,A,x,need_preprocess=True):
        if need_preprocess:
            x=blocks.Hpreprocess(x,self.input_shape)
            source_data=x[1].clone()
            tmp=x[1]
        else:
            source_data=x.clone()
            tmp=x
            
        source_embed=self.source_base_model(tmp)
        adj=[A[0],A[1],A[2]]
        source_output=self.gconv.source_forward(adj,source_embed)
        source_output=torch.stack(source_output)

        return source_data,source_output
    
    def GetTemporalOutput(self,A,x,need_preprocess=True):
        if need_preprocess:
            x=blocks.Hpreprocess(x,self.input_shape)
            source_data=x[1].clone()
            tmp=x[1]
        else:
            source_data=x.clone()
            tmp=x
        source_output=self.source_base_model(tmp)

        return source_data,source_output
    
    def passive_parameters(self):
        params=[]
        params+=self.source_base_model.parameters()
        params+=self.gconv.passive_parameters()
        return params

    def temporal_parameters(self):
        params=[]
        params+=self.source_base_model.parameters()
        return params

def compress_ts_tensor(data, time_length=6, operator='mean'):
    # time_length=2
    if operator == 'sum':
        compressed_data = data.view(data.shape[0], -1, time_length, data.shape[2], data.shape[3]).sum(dim=2)
    else:
        compressed_data = data.view(data.shape[0], -1, time_length, data.shape[2], data.shape[3]).mean(dim=2)
    return compressed_data