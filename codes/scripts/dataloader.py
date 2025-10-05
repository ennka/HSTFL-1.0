import numpy as np
import torch

from sklearn import preprocessing
import pickle
from scripts  import graphhandler,datautils
import utils


ds_path={}
#dataset='jointair/raw/data/'
def dataloader_init():
    ds_path['aqmeo']='jointair/raw/data/'
    ds_path['air']='jointair/raw/data/'
    ds_path['weather']='jointair/raw/data/'
    ds_path['NYC']='NYC/prepared/'
    ds_path['NYCBike']='NYC/prepared/'
    ds_path['NYCTaxi']='NYC/prepared/'
    ds_path['NYCT']='NYC/prepared/'
    ds_path['PEMS']='PEMS/data/'
    ds_path['PEMSFlow']='PEMS/data/'
    ds_path['PEMSSpeed']='PEMS/data/'
    ds_path['arrest']='crime/data/'
    ds_path['NYCEvent']='crime/data/'
    ds_path['Chicago']='Chicago/data/'
    ds_path['CBike']='Chicago/data/'
    ds_path['Lyon']='Lyon/processed/'
    ds_path['LyonParking']='Lyon/processed/'
  
def get_dirpath(dataset):
    if utils.check_special_field(dataset,['Lyon','LS']):
        return 'Lyon/processed/'
    if utils.check_special_field(dataset,['NYC']):
        return 'NYC/prepared/'
    if utils.check_special_field(dataset,['Chicago','CBike']):
        return 'Chicago/data/'
    if utils.check_special_field(dataset,['air','aqmeo']):
        return 'jointair/raw/data/'
    return 0


def load_file(fname,dataset):
    #print('loading', fname)
    pkl_file=open("../data/"+get_dirpath(dataset)+fname+".pkl",'rb')
    data=pickle.load(pkl_file) 
    pkl_file.close()
    return data

def print_log(metric,node_metric=None,fname="log"):
    print('Start saving experiment data!',fname)
    saved_data={'metric':metric,"node_metric":node_metric} #.tolist()
    #np.save(fname+'.npy', saved_data)
    pkl_file=open(fname+'.pkl','wb')
    pickle.dump(saved_data,pkl_file)
    pkl_file.close()


def print_attack_result(saved_data,fname="log"):
    pkl_file=open(fname+'.pkl','wb')
    pickle.dump(saved_data,pkl_file)
    pkl_file.close()

def load_data_from_file(dataset_name='aqmeo'):
    data_shapes=[]
    reshaped_data=[]
    cuts=1
    data='-1'
    partition=None
    "[[x1,y1],[x2,y2],...], shape: T,N,F"


    if dataset_name== 'aqmeo':
        data=load_file('aqmeo_data',dataset_name) # data=data[:,:,1]
        cuts=4
    elif dataset_name== 'air':
        data=load_file('aqmeo_data',dataset_name)
        cuts=4
        data=[data[0]]
    elif dataset_name=='NYC':
        data=load_file('NYC_data',dataset_name)
        partition=[14/91,14/91]
    elif dataset_name=='NYC2':
        data=load_file('NYC2_data',dataset_name)
        partition=[14/91,14/91]
    elif dataset_name=='NYCBike':
        data=load_file('NYC_data',dataset_name)
        data=[data[0]]
        partition=[14/91,14/91]
    elif dataset_name== 'Chicago':
        data=load_file('Chicago_data',dataset_name) # data=data[:,:,1]
        partition=[14/91,14/91]
    elif dataset_name== 'Chicago2':
        data=load_file('Chicago2_data',dataset_name) # data=data[:,:,1]
        partition=[14/91,14/91]
    elif dataset_name== 'Chicago3':
        data=load_file('Chicago3_data',dataset_name) # data=data[:,:,1]
        partition=[14/91,14/91]
    elif dataset_name== 'CBike':
        data=load_file('Chicago_data',dataset_name)
        data=[data[0]]
        partition=[14/91,14/91]
    elif dataset_name== 'Lyon':
        data=load_file('Lyon_data_0_0',dataset_name) # data=data[:,:,1]
    elif dataset_name== 'LyonParking':
        data=load_file('Lyon_data_0_0',dataset_name)
        data=[data[0]]

    if type(data)==str:
        data_file,_,is_MS=get_data_files(dataset_name)
        data=load_file(data_file,dataset_name)
        if not is_MS:
            data=[data[0]]

    goal_node_counts=data[0][0].shape[1]


    t=len(data)
    for i in range(t):
        x_data=data[i][0]
        y_data=data[i][1]
        #time_length=x_data.shape[0]
        
        data_shapes.append((x_data.shape,y_data.shape))
        #print((x_data.shape,y_data.shape),time_length)
        reshaped_data.append([x_data,y_data])

    return reshaped_data,data_shapes,cuts,partition,goal_node_counts


def load_data(args,device=torch.device('cpu')):
    dataset=args.dataset
    # n_his=args.n_his
    # n_pred=args.n_pred
    # batch_size=args.batch_size
    #args.adj=None
    args.data_partitions=[0.1,0.1]

    reshaped_data,args.raw_data_shapes,args.cuts,partition,args.goal_node_count=load_data_from_file(dataset)

    if partition is not None:
        args.data_partitions=partition
    data_handler=datautils.DataHandler(args,device)
    loader,args.y_scaler=data_handler.prepare_data(reshaped_data) #[[x,y]*P]
    return loader


def load_distance_file(dataset,node_count=None):
    distance='-1'

    if dataset =='aqmeo':
        distance=load_file('aqmeo_distance',dataset)

    if dataset =='air':
        distance=load_file('aqmeo_distance',dataset)
        distance=distance[:35,:35]



    if dataset =='NYC' or dataset =='NYC2':
        distance=load_file('NYC_distance',dataset)

    if dataset =='NYCBike':
        distance=load_file('NYC_distance',dataset)
        distance=distance[:node_count,:node_count]


    if dataset =='Chicago' or dataset=='Chicago2' or dataset=='Chicago3':
        distance=load_file('Chicago_distance',dataset)
    
    if dataset=='CBike':
        distance=load_file('Chicago_distance',dataset) 
        distance=distance[:node_count,:node_count]   
    
    if dataset=='LyonParking':
        distance=load_file('Lyon_distance',dataset) 
        distance=distance[:node_count,:node_count] 

    if type(distance)==str:
        _,distance_file,is_MS=get_data_files(dataset)
        distance=load_file(distance_file,dataset) 
        if not is_MS:
            distance=distance[:node_count,:node_count] 

     
    return distance

def load_gso(args):
    args.n_vertex=utils.get_n_vertex(args.raw_data_shapes)
    gh=graphhandler.GraphHandler(args=args)
    distance=load_distance_file(args.dataset,args.goal_node_count)
    args.adj=gh.generate_graph(distance)
    args.splited_distances=gh.split_matrixs(distance)
    #args.adj=gh.split_matrixs(adj)
    
def load_raw_data(args,device=torch.device('cpu')):
    dataset=args.dataset
    # n_his=args.n_his
    # n_pred=args.n_pred
    # batch_size=args.batch_size
    args.adj=None
    args.data_partitions=[0.1,0.1]

    reshaped_data,args.raw_data_shapes,args.cuts,partition,_=load_data_from_file(dataset)

    args.n_vertex=utils.get_n_vertex(args.raw_data_shapes)
    gh=graphhandler.GraphHandler(args=args)
    distance=load_distance_file(args.dataset,0)
    args.adj=gh.generate_graph(distance,normalize=False)

    return distance, reshaped_data,

def get_data_files(file_name):
    data_file,dis_file,MS_data=init_data_files()
    use_multisource_data=file_name in MS_data
    return data_file[file_name],dis_file[file_name],use_multisource_data


def init_data_files():
    MS_data=['aqmeo','NYC','Chicago','LS3','LS33']
    data_file={}
    dis_file={}
    data_file['LS3']='Lyon_data_0_42'
    dis_file['LS3']='Lyon_distance_0_42'
    data_file['LS33']='Lyon3_data_0_42'
    dis_file['LS33']='Lyon_distance_0_42'

    return data_file,dis_file,MS_data





if __name__ == "__main__":
   print('testing')
    



