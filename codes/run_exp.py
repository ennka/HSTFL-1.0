import os
import numpy  as np
import torch

import time
import pickle
#export MKL_SERVICE_FORCE_INTEL=1
DEFAULT_REPEAT_NUM=2
start_seed=4396
fn_index='_HSTFL_'

def load_file(fname="testing_file"):
    pkl_file=open(fname+".pkl",'rb')
    data=pickle.load(pkl_file) 
    pkl_file.close()
    return data['metric'],data['node_metric']

def load_attack_res(fname="logger/QFNYC_HNet_001"):
    pkl_file=open(fname+".pkl",'rb')
    data=pickle.load(pkl_file) 
    pkl_file.close()
    return data

def load_file2(fn):
    import numpy as np
    with open(fn+'.npy', 'rb') as f:
        data = np.load(f)
    return data

def save_txt(data,fn='run_exp'):
    f = open('exp_log/'+fn+fn_index+".txt",'w') 
    f.write(str(data))
    f.write('\n')
    f.close()

def format_list_numbers(lst):
    formatted_lst = []
    for num in lst:
        formatted_num = "{:.5f}".format(num)
        formatted_lst.append(formatted_num)
    return formatted_lst

m2=['SNet','HNet','FDML']

def set_local(model):
    exp_param={}
    time_param={}
    exp_param['NYCBike']=[model[0]]
    time_param['NYCBike']=[12,12]

    exp_param['air']=[model[0]]
    time_param['air']=[48,12]

    exp_param['LyonParking']=[model[0]]
    time_param['LyonParking']=[12,12]

    exp_param['CBike']=[model[0]]
    time_param['CBike']=[12,12]

    return exp_param,time_param



def exp_modules1(model): #Chicago & LS
    exp_param={}
    time_param={}
    exp_param['CBike']=[model[0]]
    time_param['CBike']=[12,12]
    exp_param['Chicago']=[model[1]]
    time_param['Chicago']=[12,12]

    exp_param['LyonParking']=[model[0]]
    time_param['LyonParking']=[12,12]
    exp_param['LS3']=[model[1]]
    time_param['LS3']=[12,12]

    return exp_param,time_param

def set_expHetero(model): #Chicago & LS
    exp_param={}
    time_param={}
    # exp_param['Chicago']=[model[1]]
    # time_param['Chicago']=[12,12]

    exp_param['LS3']=[model[1]]
    time_param['LS3']=[12,12]

    return exp_param,time_param


def set_expHN(model): 
    exp_param={}
    time_param={}
    exp_param['Chicago']=[model[1]]
    time_param['Chicago']=[12,12]

    exp_param['LS3']=[model[1]]
    time_param['LS3']=[12,12]  

    exp_param['aqmeo']=[model[1]]
    time_param['aqmeo']=[48,12]

    exp_param['NYC']=[model[1]]
    time_param['NYC']=[12,12]   

    return exp_param,time_param


def set_expFDML(model): 
    exp_param={}
    time_param={}
    exp_param['Chicago']=[model[2]]
    time_param['Chicago']=[12,12]

    exp_param['LS3']=[model[2]]
    time_param['LS3']=[12,12]  

    exp_param['aqmeo']=[model[2]]
    time_param['aqmeo']=[48,12]

    exp_param['NYC']=[model[2]]
    time_param['NYC']=[12,12]   

    return exp_param,time_param




class exp_runner():
    def __init__(self,exp=3):
        super().__init__()
        self.exp=exp
        self.exp_id=exp
        self.init_param()
    """
    Set the 'exp' to choose what to run.
    Example: set exp=1 to run experiments in local model 

    Main experiment:
    1. Local model (1)
    2. FedSim (10) & AVGsim (14) & Top1Sim (15)
    3. HSTFL-DP8 (2.1)
    4. HSTFL (2)
    5. SL-FDML (13)
    6. SL-SplitNN (7)
    
    Ablation study:
    1. HSTFL- NoPVFSRL (9)
    2. HSTFL- NoMLR (9)
    3. HSTFL- NoMA (12)
    4. HSTFL- NoVNA (4)

    Module agnostic evaluation
    1. Spatial modules (6)
    2. Temporal modules (3)

    Privacy evaluations
    1. HSTFL (-2)
    2. HSTFL-NoVNA(-4)
    3. Differential privacy: performance (8) Attack result (-8)


    Appendix:
    1. Heterogeneous data: length of time series (17) 
    2. Heterogeneous data: sampling rate (18)

    """
    def init_param(self):
        exp=self.exp
        model=m2
        if exp in [1]:
            exp_param,time_param=set_local(model)
        elif exp in [2,2.1,-2,4,-4,7,8,-8,9,10,12,14,15]:
            exp_param,time_param=set_expHN(model)
        elif exp in [3,6]:
            exp_param,time_param=exp_modules1(model)
        elif exp in [13]:
            exp_param,time_param=set_expFDML(model)
        elif exp in [17,18]:
            exp_param,time_param=set_expHetero(model)
        else:
            'ERROR'
            1/0
            return 
        self.time_param=time_param
        self.exp_param=exp_param

    def run_task(self,dataset,models,fnote='',n_history=12,n_pred=12,repeat_num=DEFAULT_REPEAT_NUM,lr=5e-3,pp=-1,local_kernel='GWN',cross_client_kernel='VNA',message_passing='all',exp_type='normal_training',cuda_index=1,structure_type='HSTFL',temporal_module='GRU',test_mla=False):
        #2/
        print("The experiment will repeat ", repeat_num,'times.')
        for m in models:
            results=[]
            node_results=[]
            WB_res=[]
            QF_res=[]
            logger=dataset+'_'+m
            for i in range(repeat_num):
                if exp_type=='normal_training':
                    log_fn=str(i)
                    
                    if test_mla:
                        os.system(f'python -u main.py --dataset {dataset} --model {m} --logfn {log_fn} --seed {i+start_seed} --n_pred {n_pred} --n_his {n_history} --lr {lr} --pp {pp} --local_kernel {local_kernel} --inter_client_kernel {cross_client_kernel} --cuda_index {cuda_index} --message_passing {message_passing} --temporal_module {temporal_module} --test_mla True')
                    else:
                        #os.system(f'python -u main.py --dataset {dataset} --model {m} --logfn {log_fn} --seed {i+start_seed} --n_pred {n_pred} --n_his {n_history} --lr {lr} --pp {pp} --local_kernel {local_kernel} --inter_client_kernel {cross_client_kernel} --cuda_index {cuda_index} --message_passing {message_passing} --temporal_module {temporal_module}')--test_param 3
                        os.system(f'python -u main.py --dataset {dataset} --model {m} --logfn {log_fn} --seed {i+start_seed} --n_pred {n_pred} --n_his {n_history} --lr {lr} --pp {pp} --local_kernel {local_kernel} --inter_client_kernel {cross_client_kernel} --cuda_index {cuda_index} --message_passing {message_passing} --temporal_module {temporal_module} ')
                    logger_fn='logger/'+dataset+'_'+m+'_'+log_fn
                    res=load_file(logger_fn)
                    results.append(res[0])
                    node_results.append(res[1])
                else:
                    log_fn=str(i)
                    """Train the model normally at first"""
                    # if pp==-1:
                    #     train_epoch=0 #This exp has already finished.
                    os.system(f'python -u main.py --dataset {dataset} --model {m} --logfn {log_fn} --seed {i+start_seed} --n_pred {n_pred} --n_his {n_history} --lr {lr} --pp {pp}  --local_kernel {local_kernel} --inter_client_kernel {cross_client_kernel} --cuda_index {cuda_index} --message_passing {message_passing} --structure_type {structure_type}')
                    logger_fn='logger/'+dataset+'_'+m+'_'+log_fn
                    res=load_file(logger_fn)
                    results.append(res[0])
                    node_results.append(res[1])

                    """White box attack"""
                    os.system(f'python -u main.py --dataset {dataset} --model {m} --logfn {log_fn} --seed {start_seed} --n_pred {n_pred} --n_his {n_history} --lr {lr} --pp {pp}   --local_kernel {local_kernel} --inter_client_kernel {cross_client_kernel} --message_passing {message_passing} --cuda_index {cuda_index} --mode inverse --structure_type {structure_type}')
                    logger_fn='logger/WB_'+dataset+'_'+m+'_'+log_fn
                    WB_res.append(load_attack_res(logger_fn).numpy())

                    """Query free attack"""
                    os.system(f'python -u main.py --dataset {dataset} --model {m} --logfn {log_fn} --seed {i+start_seed} --n_pred {n_pred} --n_his {n_history} --lr {lr} --pp {pp}   --local_kernel {local_kernel} --inter_client_kernel {cross_client_kernel} --message_passing {message_passing} --cuda_index {cuda_index} --mode train_alt --structure_type {structure_type}')
                    os.system(f'python -u main.py --dataset {dataset} --model {m} --logfn {log_fn} --seed {start_seed} --n_pred {n_pred} --n_his {n_history} --lr {lr} --pp {pp}   --local_kernel {local_kernel} --inter_client_kernel {cross_client_kernel} --message_passing {message_passing} --cuda_index {cuda_index} --mode inverse_alt --structure_type {structure_type}')
                    logger_fn='logger/QF_'+dataset+'_'+m+'_'+log_fn
                    QF_res.append(load_attack_res(logger_fn).numpy())

                
            
            #import numpy as np
            exp_result=[np.mean(results,axis=0),np.std(results,axis=0)]
            exp_result2=[np.mean(node_results,axis=0),np.std(node_results,axis=0)]
            # print(exp_result)
            # print(exp_result2,node_results.shape)
            if not exp_type=='normal_training':
                # #print(WB_res.shape,QF_res.shape)
                # print(WB_res)
                # print(QF_res)
                WB_res=[np.mean(WB_res,axis=0),np.std(WB_res,axis=0)]
                QF_res=[np.mean(QF_res,axis=0),np.std(QF_res,axis=0)]
                save_txt([WB_res,QF_res],fn=logger+fnote+'_attack_result')
            
            save_txt([exp_result,exp_result2],fn=logger+fnote)

    def run_sim(self,dataset,models,fnote='',n_history=12,n_pred=12,repeat_num=DEFAULT_REPEAT_NUM,lr=5e-3,pp=-1,local_kernel='GWN',cross_client_kernel='MGC',message_passing='all',exp_type='normal_training',cuda_index=1,structure_type='HSTFL',temporal_module='GRU',test_mla=False,no_sim=False):
        #2/0
        for m in models:
            results=[]
            node_results=[]
            WB_res=[]
            QF_res=[]
            logger=dataset+'_'+m
            for i in range(repeat_num):
                if exp_type=='normal_training':
                    log_fn=str(i)
                    os.system(f'python -u FedSim.py --dataset {dataset} --model FedSim --logfn {log_fn} --seed {i+start_seed} --n_pred {n_pred} --n_his {n_history} --lr {lr} --pp {pp} --local_kernel {local_kernel} --inter_client_kernel {cross_client_kernel} --cuda_index {cuda_index} --message_passing {message_passing} --temporal_module {temporal_module}  --test_mla {str(test_mla)} --no_sim {str(no_sim)}')
                    logger_fn='logger/'+dataset+'_FedSim_'+log_fn
                    res=load_file(logger_fn)
                    results.append(res[0])
                    node_results.append(res[1])
                else:
                    'ERROR'
                    1/0
            
            #import numpy as np
            exp_result=[np.mean(results,axis=0),np.std(results,axis=0)]
            exp_result2=[np.mean(node_results,axis=0),np.std(node_results,axis=0)]
            
            save_txt([exp_result,exp_result2],fn=logger+fnote+'FEDSIMEXP')
    

    def run_ts(self,dataset,models,fnote='',n_history=72,n_pred=72,repeat_num=DEFAULT_REPEAT_NUM,exp_type='normal_training',cuda_index=1,sampling_rate=30):
        
        for m in models:
            results=[]
            logger=dataset+'_'+m
            for i in range(repeat_num):
                if exp_type=='normal_training':
                    log_fn=str(i)
                    os.system(f'python -u ts_main.py --dataset {dataset} --sampling_rate {sampling_rate} --n_pred {n_pred} --n_his {n_history} --logfn {log_fn} --seed {i+start_seed} --cuda_index {cuda_index} ')
                    logger_fn='logger/'+dataset+'3_timeSampling_'+log_fn
                    res=load_file(logger_fn)
                    results.append(res[0])
                else:
                    'ERROR'
                    1/0
            exp_result=[np.mean(results,axis=0),np.std(results,axis=0)]
            exp_result2=[]
            
            save_txt([exp_result,exp_result2],fn=logger+fnote)
        return

    
    def run_tl(self,dataset,models,fnote='',n_history=12,n_pred=12,repeat_num=DEFAULT_REPEAT_NUM,exp_type='normal_training',cuda_index=1,tl_l=12):

        for m in models:
            results=[]
            logger=dataset+'_'+m
            for i in range(repeat_num):
                if exp_type=='normal_training':
                    log_fn=str(i)
                    os.system(f'python -u tl_main.py --dataset {dataset} --tl_l {tl_l} --logfn {log_fn} --seed {i+start_seed} --cuda_index {cuda_index} ')
                    logger_fn='logger/'+dataset+'_timeLength_'+log_fn
                    res=load_file(logger_fn)
                    results.append(res[0])
                else:
                    'ERROR'
                    1/0
            exp_result=[np.mean(results,axis=0),np.std(results,axis=0)]
            exp_result2=[]
            
            save_txt([exp_result,exp_result2],fn=logger+fnote)
        return


    def run_exp(self):
        for dataset, models in self.exp_param.items():
            print('Check args before process start')
            time.sleep(3)
            n_his,n_pred=self.time_param[dataset]
            privacy_params=[-1,64,32,16,8,4] #need update,
            lr=1e-3
            if dataset in ['aqmeo','air']:
                n_his=48
                lr=1e-4
            # elif dataset in ['CBike','Chicago']:
            #     lr=5e-3

            if self.exp_id in [1]:
                #Main results
                fnote='Local_model'
                self.run_task(dataset,models,fnote=fnote,n_history=n_his,n_pred=n_pred,lr=lr,cuda_index=1)

            if self.exp_id in [2]:
                fnote='HSTFL'
                self.run_task(dataset,models,fnote=fnote,n_history=n_his,n_pred=n_pred,lr=lr,cuda_index=0)

            if self.exp_id in [2.1]:
                fnote='HSTFL-DP'
                self.run_task(dataset,models,fnote=fnote,n_history=n_his,n_pred=n_pred,lr=lr,cuda_index=0,pp=8)
            
            if self.exp_id in [-2]:
                p=-1
                fnote='HSTFL_privacy_'+str(p)
                self.run_task(dataset,models,fnote=fnote,n_history=n_his,n_pred=n_pred,lr=lr,pp=p,exp_type='eval_security',cuda_index=0)

            if self.exp_id in [3]:
                for kernel in ['Informer','LSTM']:
                    fnote='Temporalmodule_'+kernel
                    self.run_task(dataset,models,fnote=fnote,n_history=n_his,n_pred=n_pred,lr=lr,temporal_module=kernel,cuda_index=1)

            if self.exp_id in [4]:
                for structure_type in ['simple']:
                    fnote='HSTFL-NoVNA'
                    self.run_task(dataset,models,fnote=fnote,n_history=n_his,n_pred=n_pred,lr=lr,pp=-1,cuda_index=1,structure_type=structure_type,cross_client_kernel='SLK')

            
            if self.exp_id in [-4]:
                #SoftLinkage VFL Privacy baseline
                for structure_type in ['simple']:
                    fnote='HSTFL-NoVNA_privacy'
                    self.run_task(dataset,models,fnote=fnote,n_history=n_his,n_pred=n_pred,lr=lr,pp=-1,exp_type='eval_security',cuda_index=1,structure_type=structure_type,cross_client_kernel='SLK')

            if self.exp_id in [6]:
                #Local spatial kernel
                for kernel in ['DiffusionConv','GCN','Transformer']:
                    fnote='Spatialmodule_'+kernel
                    self.run_task(dataset,models,fnote=fnote,n_history=n_his,n_pred=n_pred,lr=lr,local_kernel=kernel,cuda_index=0)

            if self.exp_id in [7]:
                for kernel in ['SLK']:
                    fnote='SL-splitNN'
                    self.run_task(dataset,models,fnote=fnote,n_history=n_his,n_pred=n_pred,lr=lr,message_passing='first',cross_client_kernel=kernel,cuda_index=1)

            if self.exp_id in [8]:
                for p in privacy_params[1:]:
                    fnote='HSTFL-DPs_'+str(p)
                    self.run_task(dataset,models,fnote=fnote,n_history=n_his,n_pred=n_pred,lr=lr,pp=p,cuda_index=1)
            
            if self.exp_id in [-8]:
                for p in privacy_params[1:]:
                    fnote='HSTFL-DPs_privacy_'+str(p)
                    self.run_task(dataset,models,fnote=fnote,n_history=n_his,n_pred=n_pred,lr=lr,pp=p,exp_type='eval_security',cuda_index=1)

            if self.exp_id in [9]:
                for message_passing in ['first','last']:
                    if message_passing=='first':
                        fnote='HSTFL-NoPVFSRL'
                    else:
                        fnote='HSTFL-NoMLR'
                    self.run_task(dataset,models,fnote=fnote,n_history=n_his,n_pred=n_pred,lr=lr,message_passing=message_passing,cuda_index=0)
            
            if self.exp_id in [12]:
                for test_mla in [True]:
                    fnote='HSTFL-NoMA'
                    self.run_task(dataset,models,fnote=fnote,n_history=n_his,n_pred=n_pred,lr=lr,test_mla=test_mla,cuda_index=0)

            if self.exp_id in [13]:
                fnote='SL-FDML'
                self.run_task(dataset,models,fnote=fnote,n_history=n_his,n_pred=n_pred,lr=lr,cuda_index=0)

            if self.exp_id in [10]:
                fnote='FedSim'
                self.run_sim(dataset,models,fnote=fnote,n_history=n_his,n_pred=n_pred,lr=lr,cuda_index=0,)

            if self.exp_id in [14]:    
                for kernel in ['SLK']:
                    fnote='AVGSim'
                    self.run_sim(dataset,models,fnote=fnote,n_history=n_his,n_pred=n_pred,lr=lr,cuda_index=0,cross_client_kernel=kernel,no_sim=True)
            
            if self.exp_id in [15]:
                for kernel in ['SL1']:
                    fnote='Top1Sim'
                    self.run_sim(dataset,models,fnote=fnote,n_history=n_his,n_pred=n_pred,lr=lr,cuda_index=1,cross_client_kernel=kernel,no_sim=True)
            
            if self.exp_id in [17]: 
                fn='Heterogenoeus_time_length'

                if dataset=='Chicago':
                    time_lengths=[2,3,6,12,24,48]
                else:
                    time_lengths=[4,8,12,16,20,24]

                for time_length in time_lengths:
                    fnote=fn+str(time_length)
                    self.run_tl(dataset,models,fnote=fnote,n_history=-1,tl_l=time_length,cuda_index=0)


            if self.exp_id in [18]: 
                fn='Heterogenoeus_sampling_rate'
                time_samplings=[5,10,15,20,30,60]
                if dataset=='LS3':
                    n_history,n_pred=36,36
                else:
                    n_history,n_pred=72,72
                for time_sampling in time_samplings:
                    fnote=fn+str(time_sampling)
                    self.run_ts(dataset,models,fnote=fnote,n_history=n_history,n_pred=n_pred,sampling_rate=time_sampling,cuda_index=1)
                    #parser.add_argument('--sampling_rate', type=int, default=30) 



def start_exp():

    hanlder=exp_runner(18)

    hanlder.run_exp()
    print('exp finish')
    return 0

if __name__ == "__main__":
    np.set_printoptions(suppress=True, formatter={'float': '{:0.5f}'.format})
    start_exp()
    print('finish')
