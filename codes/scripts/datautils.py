import numpy as np
import torch
from torch.utils.data import Dataset,TensorDataset,DataLoader


class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, mean=0.1, std=0.1):
        self.mean = mean
        self.std = std
        self.valid=True
        #print('Scaler is invalid!!!!!')

    def fit (self,data):
        self.mean=np.mean(data,axis=(0,1))
        self.std=np.std(data,axis=(0,1))

    def to_device(self,device):
        self.mean=torch.tensor(self.mean).to(device)
        self.std=torch.tensor(self.std).to(device)
    

    def fit_transform(self,data):

        self.fit(data=data)
        return self.transform(data=data)

    def transform(self, data):
        if not self.valid:
            return data
            
        #print(data.shape)
        return (data - self.mean) / (self.std+1e-9)

    def inverse_transform(self, data):
        # if torch.is_tensor(data) and data.is_cuda:
        #     d=data.cpu().detach().numpy() 
        return self._inverse_transform(data)
    
    def _inverse_transform(self, data):
        return (data * self.std) + self.mean

    def batch_transform(self,data):
        return [self.transform(data[i]) for i in range(len(data))]
    

    def show(self):
        return self.mean,self.std


class DataHandler():
    def __init__(self,args,device=torch.device('cpu'),dynamic=False,pretrain=False,val=0.1,test=0.1):
        self.n_his=args.n_his
        self.n_pred=args.n_pred
        self.cuts=args.cuts
        self.batch_size=32 #args.batch_size
        self.device=device
        self.dynamic=dynamic
        self.pretrain=pretrain
        self.dataset=args.dataset
        self.val_size=val#14/91
        self.test_size=test#14/91
        self.shape=args.raw_data_shapes
    
    def merge_data(self,data):
        merged_data=[]
        for p in range(len(data)):
            #print(data[p].shape)
            cut_data=np.concatenate(data[p],axis=0) 
            merged_data.append(cut_data)
        return merged_data
    

    def data_split(self,data): #Cut dataset
        length=data.shape[0]
        val_len=int(np.ceil(length*self.val_size))
        test_len=int(np.ceil(length*self.test_size))
        train_len=length-val_len-test_len
        return data[:train_len],data[train_len:train_len+val_len],data[train_len+val_len:train_len+val_len+test_len]
    
    def cut_and_split(self,data,cuts=12,dim=1000):
        # print('dim is', dim)
        length=data.shape[0]
        cut_lens=[]
        for i in range(cuts-1):
            cut_lens.append(int(length/cuts))
        cut_lens.append(length-sum(cut_lens))
        train_data,val_data,test_data=[],[],[]
        index=0
        for i in range(cuts):
            data_for_split=data[index:index+cut_lens[i]]
            index+=cut_lens[i]
            train,test,val=self.data_split(data_for_split)
            train_data.append(train[:,:dim])
            val_data.append(val[:,:dim])
            test_data.append(test[:,:dim])
        return train_data,val_data,test_data

    def norm_split(self,data,is_target=False):
        train,val,test=[],[],[]
        if is_target:
            y_shape=self.shape[0][1]
            y_dim=y_shape[1]*y_shape[2]
        else:
            y_dim=1000
        
        participant_num=len(data)
        for i in range(len(data)):#type数量
            t,v,e=self.cut_and_split(data[i],self.cuts,dim=y_dim)
            train.append(t)
            val.append(v)
            test.append(e)
    
        """
        train: 
        0 dim- type of data (participants)
        1 dim- cut data for different period
        2 dim -sample of data
        """


        P=1 if is_target else participant_num 

        merged_train=self.merge_data(train)
        # val=self.merge_data(val)
        # test=self.merge_data(test)
        """
        merged_train: 
        0 dim- type of data (participants)
        1 dim -sample of data
        """
        train_data,val_data,test_data=[],[],[]
        zscores=[]
        for p in range(P):
            zscore = StandardScaler()#preprocessing.StandardScaler()
            zscore.fit(merged_train[p])
            # print(merged_train[p].shape,zscore.mean.shape)
            # 1/0
            zscores.append(zscore)
            if is_target:
                train_data.append(train[p])
                val_data.append(val[p])
                test_data.append(test[p])
            else:
                train_data.append(zscore.batch_transform(train[p]))
                val_data.append(zscore.batch_transform(val[p]))
                test_data.append(zscore.batch_transform(test[p]))
            
        train_data,val_data,test_data=self.flatten_data(train_data),self.flatten_data(val_data),self.flatten_data(test_data)

        return zscores,[train_data,val_data,test_data]
    
    def scale_hetero_data(self,data,scaler):
        res=[]
        for i in range(self.cuts):
            hetero_data=[]
            for p in range(len(data)):
                hetero_data.append(data[p][i])
            hetero_data=np.concatenate(hetero_data,axis=1)
            #print(hetero_data.shape)
            res.append(scaler.transform(hetero_data))
        
        return res
    
    def prepare_dataloader(self,xdata,ydata,shuffle=True,is_test=False): #data: dict        
       
        x_data,y_data=[],[]
        for i in range(self.cuts):
            x_cut,y_cut=self.prepare_batch_data(xdata[i],ydata[i])
            x_data.append(x_cut)
            y_data.append(y_cut)
        x_data=torch.cat(x_data,dim=0)
        y_data=torch.cat(y_data,dim=0)

        #print(x_data.shape,y_data.shape)
        data_ = TensorDataset(x_data,y_data)      
        data_iter = DataLoader(dataset=data_, batch_size=self.batch_size, shuffle=shuffle)
        return data_iter
    
    def prepare_batch_data(self,xdata,ydata):
        length=xdata.shape[0]-self.n_his-self.n_pred
        x_data=[]
        y_data=[]
        for i in range(length):
            head = i
            tail = i + self.n_his
            x_data.append(xdata[head:tail])
            y_data.append(ydata[tail:tail+self.n_pred])
        #print(np.array(y_data).shape,length)
        x_data=torch.tensor(np.array(x_data), dtype=torch.float32).to(self.device)
        y_data=torch.tensor(np.array(y_data).reshape(length,-1), dtype=torch.float32).to(self.device)
        return x_data,y_data
    
    def get_loader(self):
        train_loader=self.prepare_dataloader(self.x_data[0],self.y_data[0]) #可能会用这一部分数据pretrain autoencoder
        #print('train loader finish')
        val_loader=self.prepare_dataloader(self.x_data[1],self.y_data[1])
        #print('val loader finish')
        test_loader=self.prepare_dataloader(self.x_data[2],self.y_data[2],shuffle=False)
        #print('test loader finish')
        pretrain_loader=0
        preval_loader=0
        # if self.pretrain:
        #     print(self.pertrain_Tdata.shape)
        #     pretrain_loader=self.get_pretrain_data(self.pertrain_Tdata) 
        #     preval_loader=self.get_pretrain_data(self.pertrain_Vdata)

        """loader: {[x1,x2,x3],[y1,y2,y3]}"""
        loader=[train_loader,val_loader,pretrain_loader,preval_loader,test_loader]
        #loader=[train_loader,val_loader,pretrain_loader,preval_loader,val_loader]
        return loader
    
    def flatten_data(self,data):
        cutted_flatten_data=[]
        #print(len(data),len(data[0])) 
        for c in range(len(data[0])):#P
        #print('start flatten')
            d=[]
            for p in range(len(data)):#C
                tmp=data[p][c].reshape(data[p][c].shape[0],-1)
                d.append(tmp) #T, N*F 
                #print(tmp.shape)
            d=np.concatenate(d,axis=1)
            cutted_flatten_data.append(d)
        #print(d.shape)
        return cutted_flatten_data # cuts of data(from all P)

    def prepare_data(self,data):
        "Format [[x1,y1],[x2,y2],...], shape: T,N,F"
        x_data=[]
        y_data=[]
        for i in range(len(data)):
           x_data.append(data[i][0])
           y_data.append(data[i][1])
        self.x_zscores,x_data=self.norm_split(x_data)
        self.y_zscores,y_data=self.norm_split(y_data,is_target=True)


        self.x_data=x_data
        self.y_data=y_data
        loader=self.get_loader()
        return loader,self.y_zscores[0]
    
    def shape_back(self,data,is_label=False):
        if is_label:
            B=data.shape[0]
            data_shape=self.shape[0][1]
            N,F=data_shape[1],data_shape[2]
            shaped_data=data.reshape(B,-1,N,F)
            return shaped_data

        data_shape=self.shape[0][0],self.shape[1][0]
        #print(self.shape)
        #l=2
        multiparty_data=[]
        index=0
        B,T=data.shape[0],data.shape[1]
        for p in range(2):
            N,F=data_shape[p][1],data_shape[p][2]
            input_f=N*F
            shaped_data=data[:,:,index:index+input_f].reshape(B,T,N,F)
            multiparty_data.append(shaped_data)
            index+=input_f
        return multiparty_data

    def get_data_from_loader(self,loader):
        all_x=[]
        all_y=[]
        for batch_data in loader:
            all_x.append(batch_data[0])
            all_y.append(batch_data[1])
  
        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0)
        all_x=self.shape_back(all_x)
        all_y=self.shape_back(all_y,is_label=True)
        return all_x,all_y


if __name__ == "__main__":
    T,N=100,10
    list_x=[np.ones((T,N)),2*np.ones((T,N)),3*np.ones((T,N))]
    x=np.array(list_x)
    x=np.transpose(x,(1,2,0))
    y=torch.tensor(x)
    z=y.to('cuda')
    scaler=StandardScaler()
    scaler.fit(x) #T,N,F
    extended_x=np.zeros((32,12,10,3)) #B,seq_len,N,F
    print(extended_x.shape)
    print(x.shape)
    x_new=scaler.inverse_transform(extended_x)
    print(x_new.shape)
    print(np.mean(x_new[...,0]))
    print(np.mean(x_new[...,1]))
    print(np.mean(x_new[...,2]))