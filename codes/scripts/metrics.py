import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import torch
import torch.nn as nn
import scripts.loss_ as loss_
from sklearn.metrics import mean_absolute_percentage_error
from utils import get_input_dim,get_output_dim,get_n_vertex,get_output_shape,get_out_vertex,compress_ts

def calculate_full_metrics(args,model,loader):
    default_metrics=[loss_.masked_mae,loss_.masked_rmse,loss_.masked_smape]
    metric_loss=np.array([0.0]*len(default_metrics))
    model.eval()
    test_num = 0
    adj=args.adj
    n_pred=args.n_pred
    N=get_out_vertex(args.raw_data_shapes)#get_output_dim(args.raw_data_shapes,1)
    node_metrics=np.array([[0.0]* N]*len(default_metrics),dtype=float)
    with torch.no_grad():
        for unit in loader:
            data, target=unit
            B=target.shape[0]
            pred=model(A=adj,x=data)      
            predict = args.y_scaler.inverse_transform(pred.reshape(B,n_pred,N,-1)) #B,T,N,F
            target=target.reshape(B,n_pred,N,-1)

            try:

                target=compress_ts(target,time_length=args.tl,operator=args.ts_op) #Compress time series of the target
                predict=compress_ts(predict,time_length=args.tl,operator=args.ts_op) #Compress time series of the active party
                #print('Handling TS')
            except Exception as inst:
                # print('>LJLJKLLJLKj')
                # print(inst)
                pass
            
            test_num+=B
            for i in range(len(default_metrics)):
                m=0
                for j in range(N):
                    node_value=default_metrics[i](predict[:,:,j], target[:,:,j],0).item() * B
                    node_metrics[i,j]+=node_value
                    m+=node_value
                metric_loss[i]=metric_loss[i]+m/N

    
    metric_loss=metric_loss/test_num
    node_metrics=node_metrics/test_num

    return metric_loss,node_metrics
