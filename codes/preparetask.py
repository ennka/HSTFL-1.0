import torch
import argparse
from models import models_
from utils import get_input_dim,get_output_dim,get_n_vertex,get_in_fn,get_out_fn,get_input_shape


#model=models.LSTM(input_dim=input_dim,hidden_dim=256,n_vertex=n_vertex).to(device)

def get_model(args):
    model=args.model
    dataset=args.dataset

    ds=args.raw_data_shapes
    n_pred=args.n_pred

    input_dim=get_input_dim(ds)
    output_dim=get_output_dim(ds,n_pred)
    args.output_dim=output_dim

    input_shape=get_input_shape(ds)
    args.input_shape=input_shape
    in_fn,out_fn=get_in_fn(ds),get_out_fn(ds)

    # print(ds,output_dim,args.n_vertex)
    # 1/0
    #n_vertex=get_n_vertex(args.raw_data_shapes)
    l=args.layer_num
    kernel=[args.local_kernel,args.inter_client_kernel]
    if model=='HNet':
        model=models_.Hnet(in_dim=in_fn,hid_dim=64,out_dim=out_fn,n_pred=args.n_pred,input_shape=input_shape,protection_param=args.pp,kernel=kernel,message_passing=args.message_passing,temporal_module=args.temporal_module,layer_num=l,test_mla=args.test_mla)
    elif model=='SNet':
        model=models_.Snet(in_dim=in_fn,hid_dim=64,out_dim=out_fn,n_pred=args.n_pred,input_shape=input_shape,kernel=kernel,temporal_module=args.temporal_module,layer_num=l)
    elif model=='FDML':
        model=models_.SLFDML(in_dim=in_fn,hid_dim=64,out_dim=out_fn,n_pred=args.n_pred,input_shape=input_shape,protection_param=args.pp,kernel=kernel,message_passing=args.message_passing,temporal_module=args.temporal_module,layer_num=l,test_mla=args.test_mla)
    elif model=='FedSim':
        model=models_.FedSim(in_dim=in_fn,hid_dim=64,out_dim=out_fn,n_pred=args.n_pred,input_shape=input_shape,protection_param=args.pp,kernel=kernel,message_passing=args.message_passing,temporal_module=args.temporal_module,layer_num=l,test_mla=args.test_mla,arg=args)
    else:
        print("ERROR")
        1/0

    return model

