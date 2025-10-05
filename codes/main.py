import os

import numpy
import torch
import numpy as np


import argparse
import random
import torch.nn as nn

from models import models_
import torch.optim as optim
#import torch.utils as utils
import utils
import time

from scripts import metrics,earlystopping,dataloader,loss_
import preparetask

from security.attacks import get_source_embeddings,inverse_data_samples,eval_attacks,sample_time_series


datasets=['aqmeo','air','NYC','NYCBike','CBike','Chicago','PEMS','PEMSFlow','NYCT','NYCTaxi','LyonParking','LS3']

def set_env(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_args():
    parser = argparse.ArgumentParser(description='HSTFL')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--cuda_index', type=int, default=1, choices=[0,1])
    parser.add_argument('--seed', type=int, default=4396, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--mode', type=str, default='train', help='train / train_alt / inverse /inverse_alt')
    parser.add_argument('--structure_type', type=str, default='HSTFL', help='HSTFL/simple')
    parser.add_argument('--dataset', type=str, default='Chicago', choices=datasets)#PEMS[Flow,Speed]
    parser.add_argument('--graph_method', type=str, default='threshold', choices=['threshold','exp','inverse'])
    parser.add_argument('--pp', type=float, default=-1)
    parser.add_argument('--n_his', type=int, default=12)
    parser.add_argument('--n_pred', type=int, default=12, help='the number of time interval for predcition, default as 6')
    parser.add_argument('--model', type=str, default='HNet', choices=['SNet','HNet','FDML'])
    parser.add_argument('--droprate', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning HSTFL/codes2/main.pyrate')
    parser.add_argument('--weight_decay_rate', type=float, default=1e-4, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=32)    
    parser.add_argument('--epochs', type=int, default=250, help='epochs, default as 10000')
    parser.add_argument('--opt', type=str, default='adam', help='optimizer, default as adam')
    parser.add_argument('--step_size', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.90)
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--logfn', type=str, default='testing', help='File name')
    parser.add_argument('--local_kernel', type=str, default='GWN', choices=['DiffusionConv','GWN','GCN','Transformer']) #GWN(default), GCN, DiffusionConv, Transformer
    parser.add_argument('--temporal_module', type=str, default='GRU', choices=['GRU','GDCC','Informer','LSTM'])
    parser.add_argument('--inter_client_kernel', type=str, default='VNA', choices=['VNA','SL1','SLK'])
    parser.add_argument('--test_mla', type=bool, default=False)
    parser.add_argument('--message_passing', type=str, default='all', choices=['first','all','last'])
    parser.add_argument('--layer_num', type=list, default=[1,3])
    args = parser.parse_args()
    
    print('Training configs: {}'.format(args))
    set_env(args.seed)

    if args.cuda_index==1:
            print('set cuda to 1')
            os.environ['CUDA_VISIBLE_DEVICES']='1'
    else:
            print('set cuda to 0')
            os.environ['CUDA_VISIBLE_DEVICES']='0'
    
    if args.enable_cuda and torch.cuda.is_available():
        #os.environ['CUDA_VISIBLE_DEVICES']='1'
        
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.autograd.set_detect_anomaly(True)
    return args,device

def prepare_data(args,device):
    dataloader.dataloader_init()
    loader=dataloader.load_data(args,device)
    dataloader.load_gso(args)
    args.adj=[torch.tensor(args.adj[i]).to(device) for i in range(len(args.adj))]
    return loader

def prepare_model(args,device):
    loss = loss_.masked_mae
    #loss=
    es = earlystopping.EarlyStopping(mode='min', min_delta=0.0, patience=args.patience,model=args.model,dataset=args.dataset,cuda_index=args.cuda_index)
    model=preparetask.get_model(args)
    model=model.to(device)
    print(model.parameters(),device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate,)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    args.y_scaler.to_device(device)
    

    return loss,es,model, optimizer, scheduler

def train(args, loader, loss, es, model, optimizer, scheduler):
    #return model
    print('data status',args.raw_data_shapes,args.adj[0].shape)
    adj=args.adj
    train_loader=loader[0]
    val_loader=loader[1]
    train_loss,train_num=0.0,0
    n_pred=args.n_pred
    N=utils.get_out_vertex(args.raw_data_shapes)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    val_loss=val(model,val_loader,loss,args)
    print('start training with loss',val_loss)
    es.init(model)
        #    num_params = sum([param.numel() for param in  model.parameters()])
        # print(num_params)
    #return model
    train_epoch_len=len(train_loader)
    #print(train_epoch_len)

    for i in range(args.epochs):
        model.train()
        #train_loader.shuffle()
        train_num=0
        t1 = time.time()
        #count=0
        for batch_idx, unit in enumerate(train_loader):
            t2=time.time()
            data,target=unit[0],unit[1]#data: B,T,P {N*F}
            #print(data.shape)
            #target=data
            B=data.shape[0]
            pred=model(A=adj,x=data)
            p=pred.reshape(B,n_pred,N,-1)
            
            predict = args.y_scaler.inverse_transform(p) #B,T,N,F
            target=target.reshape(B,n_pred,N,-1)
            #print(p.shape,target.shape)
            # print(pred)
            # print(predict.shape)
            batch_loss = loss(predict, target,0)
            #print(batch_loss)
            # 1/0
            optimizer.zero_grad()                   # clear gradients for this training step
            batch_loss.backward()                       # backpropagation, compute gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            train_loss+=batch_loss.item() * B
            train_num+=B
            #print('time of batch',time.time()-t2,batch_idx)
            #count+=1
        # print(count)
        # 1/0
        scheduler.step()
        train_loss=train_loss/train_num   
        val_loss=val(model,val_loader,loss,args)
        print('train &val loss at iteration',i ,' : ' ,train_loss,val_loss.item())#,time.time()-t1 )
        #print('training time',time.time()-t1)

        if es.step(val_loss,model):
            print("stop at iteration", i)
            return es.get_model(model)
        
    print("stop at max iteration", args.epochs)
    return es.get_model(model)

def val(model,loader,loss,args):
    adj=args.adj
    n_pred=args.n_pred
    N=utils.get_out_vertex(args.raw_data_shapes)
    model.eval()
    val_loss,val_num=0.0,0
    for unit in loader:   
        data,target=unit[0],unit[1]
        B=data.shape[0]
        pred=model(A=adj,x=data)
        predict = args.y_scaler.inverse_transform(pred.reshape(B,n_pred,N,-1)) #B,T,N,F
        target = target.reshape(B,n_pred,N,-1)
        batch_loss=loss(predict, target,0)

        val_loss+=batch_loss.item() * data.shape[0]
        val_num+=data.shape[0]
    return torch.tensor(val_loss/val_num)

def test(args,test_loader,loss=None, model=None):
    metric,node_metric=metrics.calculate_full_metrics(args,model,test_loader)
    dataset=args.dataset
    print("Dataset",dataset,",\n Test MAE ",metric[0],",| RMSE ", metric[1]," | Test SMAPE ",metric[2])
    logger_info='logger/'+args.dataset+'_'+args.model+'_'+args.logfn
    dataloader.print_log(metric,node_metric,fname=logger_info)
    return 0

def train_alternative(args, loader, loss, es, model, optimizer, scheduler,structure_type='HSTFL'):
    print('Train alternative',args.raw_data_shapes,args.adj[0].shape)
    model=es.get_model(model)
    es.set_alt_path() #switch the training method into ALT mode

    adj=args.adj
    train_loader=loader[0]
    val_loader=loader[1]
    train_loss,train_num=0.0,0
    n_pred=args.n_pred
    N=utils.get_out_vertex(args.raw_data_shapes)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    val_loss=val(model,val_loader,loss,args)
    print('Load model with loss',val_loss)
    print('Reset model parameters')
    #To simulate the case that the passive party get training data but no model
    if structure_type=='HSTFL':
        model.reset_passive()
        param=model.passive_parameters()
    else:
        model.reset_temporal()
        param=model.temporal_parameters()

    val_loss=val(model,val_loader,loss,args)
    print('start training with loss',val_loss)
    optimizer = optim.Adam(param, lr=args.lr, weight_decay=args.weight_decay_rate,)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    es.init(model)
    #val_loss=val(model,val_loader,loss,args)
    
    for i in range(args.epochs):
        model.train()
        #train_loader.shuffle()
        train_num=0
        t1 = time.time()
        #count=0
        for batch_idx, unit in enumerate(train_loader):
            t2=time.time()
            data,target=unit[0],unit[1]#data: B,T,P {N*F}
            #print(data.shape)
            B=data.shape[0]
            pred=model(A=adj,x=data)
            
            predict = args.y_scaler.inverse_transform(pred.reshape(B,n_pred,N,-1)) #B,T,N,F
            target=target.reshape(B,n_pred,N,-1)
            # print(pred)
            # print(predict.shape)
            batch_loss = loss(predict, target,0.0)
            # print(batch_loss)
            # 1/0
            optimizer.zero_grad()                   # clear gradients for this training step
            batch_loss.backward()                       # backpropagation, compute gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            train_loss+=batch_loss.item() * B
            train_num+=B

        scheduler.step()
        train_loss=train_loss/train_num   
        val_loss=val(model,val_loader,loss,args)
        print('train &val loss at iteration (ALT)',i ,' : ' ,train_loss,val_loss.item())#,time.time()-t1 )
        #print('training time',time.time()-t1)
        if es.step(val_loss,model):
            print("stop at iteration", i)
            return es.get_model(model)
        
    print("stop at max iteration", args.epochs)
    return es.get_model(model)

def inverse_data(args, loader, loss, es,model, use_alt=False,structure_type='HSTFL'):
    model=es.get_model(model)
    for param in model.parameters():
        param.requires_grad = False
    
    train_loader=loader[0]
    val_loader=loader[1]
    test_loader=loader[-1]
    val_loss=val(model,val_loader,loss,args)
    print('Checking model before attack: val loss',val_loss)
    #step 1 generated source embeddings
    target_data,source_embeddings=get_source_embeddings(model,test_loader,args,structure_type=structure_type)
    print('Finish generating source embeddings')
    target_size=target_data[0][0].size()
    #Step 2 using k-means to sample time-series to perform attack
    indexs=sample_time_series(target_data)
    result=[]
    fn=str(args.dataset)+'_WB'

    if use_alt:
        fn=str(args.dataset)+'_QF'
        es.set_alt_path()
        model=es.get_model(model)
        for param in model.parameters():
            param.requires_grad = False
            #print('1')

    for id in indexs:
        #Step 3  inverse the data samples
        reconstructed_data,data_indexs=inverse_data_samples(model,source_embeddings,args,target_size,data_indexs=id,structure_type=structure_type)
        #select a part of data to invese 
        #Step 4 evaluate the result
        r=eval_attacks(target_data,reconstructed_data,data_indexs,scaler=args.y_scaler,id=id,fn=fn)
        #eval_attacks(target_data,reconstructed_data,data_indexs)
        #[2,4]
        result.append(torch.tensor(r))
    #print(result)
    result=torch.mean(torch.stack(result),dim=0)
    print(result.shape)
    attack_type='WB'
    if use_alt:
        attack_type='QF'
    
    logger_info='logger/'+attack_type+'_'+args.dataset+'_'+args.model+'_'+args.logfn
    dataloader.print_attack_result(result, fname=logger_info)

    

    return result

def main():
    args,device=get_args()
    loader=prepare_data(args,device)
    if args.mode=='train':
        loss, es, model, optimizer, scheduler = prepare_model(args,device)
        model=train(args, loader[:], loss, es, model, optimizer, scheduler)
        print('start testing')
        test(args, loader[-1],loss, model)
        return 0
    elif args.mode=='train_alt':
        loss, es, model, optimizer, scheduler = prepare_model(args,device)
        model=train_alternative(args, loader[:-1], loss, es, model, optimizer, scheduler,structure_type=args.structure_type)
        print('train_alt finish')
        return 1
    elif args.mode=='inverse':
        loss, es, model, optimizer, scheduler = prepare_model(args,device)
        result=inverse_data(args, loader,loss, es,model,structure_type=args.structure_type)
        print('inverse result')
        print(result)
        return 2
    elif args.mode=='inverse_alt':
        loss, es, model, optimizer, scheduler = prepare_model(args,device)
        result=inverse_data(args, loader,loss, es,model,use_alt=True,structure_type=args.structure_type)
        print('inverse alt result')
        print(result)
        return 2
    return -1

if __name__ == "__main__":

    t1=time.time()

    result=main()
    print('time consumption', time.time()-t1)
    print('return code:',result)
    #z=dataloader.load_file()

