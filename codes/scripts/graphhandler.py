import numpy as np
from utils import save_file
import heapq
import pickle
import scipy.sparse as sp
from scipy.sparse import linalg
import os

adj_parameters={}
def init_params():
    adj_parameters['aqmeo']=12
    adj_parameters['air']=12
    adj_parameters['weather']=12
    adj_parameters['NYC']=1
    adj_parameters['NYCBike']=1
    adj_parameters['NYCTaxi']=2
    adj_parameters['PEMS']=2
    adj_parameters['PEMSSpeed']=2
    adj_parameters['PEMSFlow']=2

def threshold(distance,value, is_small=True):
    if is_small:
        return distance<value
    else:
        return distance>value

def knearest(distance,k=2):
    res=np.zeros(distance.shape)
    l=distance.shape[0]
    for i in range(l):
        d=distance[i,:]
        v=np.max(heapq.nsmallest(k+1,d))
        res[i,:]=(d<=v)
    return res

def check_density(matrix):
    res=1-np.sum(matrix==0)/np.size(matrix)
    print('avg connect',res)
    print('connect/node',res*matrix.shape[0])
    return res,res*matrix.shape[0]

def Identity(distance):
    l=distance.shape[0]
    adj=np.zeros(distance.shape)
    for i in range(l):
        adj[i,i]=1
    return adj

def threshold_max(distance,max=5):
    res_matrix=np.zeros(distance.shape)
    res_matrix[:]=np.inf
    for i in range(distance.shape[0]):
        for j in range(distance.shape[1]):
            if distance[i,j]<max:
                res_matrix[i,j]=distance[i,j] 
    return res_matrix

def threshold_gaussian(distance,value=12):
    d=threshold_max(distance,25)
    dist = d[~np.isinf(d)].flatten()
    std=dist.std()
    print('std',std)
    gaussian=np.exp(-2*np.square(distance/std))
    thres=threshold(gaussian,0.1,is_small=False)
    return gaussian*thres

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(axis=1))
    #print(d.shape)
    #print("?")
    d_inv_sqrt = np.power(d, -0.5).flatten()

    # print(d)
    
    # 1/0
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    #print(type(normalized_laplacian))
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def check_std(distance):
    std=distance.std()
    res=(distance<std).astype(np.float64)
    return np.sum(res,axis=1)

class GraphHandler():
    def __init__(self,args):
        self.args=args
        self.P=len(self.args.n_vertex)
        self.method=self.knearest#self.Random#self.Identity#self.knearest#Identity#
        self.method2=self.Identity
        self.adj_filename='processed/'+args.dataset+'_'+args.model+'_adj.pkl'
        self.use_file=os.path.exists(self.adj_filename)
        self.init_params()
        self.local_k=4
        self.cross_client_k=4 if not args.inter_client_kernel in ['Top1Sim','SL1'] else 0 #k+1 for KNN
        
    def init_params(self):
        "Invalid"
        adj_parameters={}
        adj_parameters['aqmeo']=12
        adj_parameters['air']=12
        adj_parameters['weather']=12
        adj_parameters['NYC']=1
        adj_parameters['NYCBike']=1
        adj_parameters['NYCTaxi']=2
        adj_parameters['NYCT']=2
        adj_parameters['PEMS']=3
        adj_parameters['PEMSSpeed']=3
        adj_parameters['PEMSFlow']=3
        adj_parameters['arrest']=3
        adj_parameters['NYCEvent']=3
        adj_parameters['Chicago']=3
        adj_parameters['CBike']=3
        adj_parameters['Lyon']=3
        adj_parameters['LyonParking']=3
        self.adj_parameters=adj_parameters
    
    def generate_graph(self,distance,reload=True,normalize=True):
        if self.use_file and reload==False:
            self.load_adj()
            return self.adj
        value=self.adj_parameters[self.args.dataset]  if self.args.dataset in self.adj_parameters.keys() else 0
        adj=self.method(distance,value=value).astype(np.float32)

        if normalize:
            adj=self.normalize(adj)
        self.adj=self.split_matrixs(adj)

        self.save_adj()
        return self.adj
    
    def split_matrixs(self,matrix):
        if self.P==1:
            res=[matrix]
        else:
            n0=self.args.n_vertex[0]
            res=[matrix[:n0,:n0],matrix[:n0,n0:],matrix[n0:,n0:],matrix[n0:,:n0]] #A->A, B->A, B->B, A->B 
        return res
    
    def merge_matrixs(self,matrix):
        if self.P==1:
            return matrix[0]
        else:
            m1=np.concatenate((matrix[0],matrix[1]),axis=1)
            m2=np.concatenate((matrix[2],matrix[3]),axis=1)
            return np.concatenate((m1,m2),axis=0)
        
    def normalize(self,adj):
        norm_adj=calculate_normalized_laplacian(adj)
        norm_adj=norm_adj.astype(np.float32).todense()
        return norm_adj
    
    def save_adj(self):
        pkl_file=open(self.adj_filename,'wb')
        pickle.dump(self.adj,pkl_file)
        pkl_file.close()

    def load_adj(self):
        pkl_file=open(self.adj_filename,'rb')
        self.adj=pickle.load(pkl_file)
        pkl_file.close() 
    
    def threshold_gaussian(self,distance,value=0):
        return threshold_gaussian(distance)
    
    def threshold(self,distance):
        return threshold(distance)
    
    def Identity(self,distance,value=0):
        return Identity(distance)
    
    def Random(self,distance,value=0):
        z=np.random.rand(*distance.shape)
        return z
    
    def knearest(self,distance,k=4,value=10):
        split_distance=self.split_matrixs(distance)
        ks=[self.local_k,self.cross_client_k,self.local_k,self.cross_client_k]
        adjs=[knearest(split_distance[i],ks[i]) for i in range(len(split_distance))]
        #for i in range()
        
        return self.merge_matrixs(adjs)



