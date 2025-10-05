import pickle
import numpy as np
import math

def save_file(data,fname="testing_file"):
    pkl_file=open(fname+".pkl",'wb')
    pickle.dump(data,pkl_file)
    pkl_file.close()

def load_file(fname="testing_file"):
    pkl_file=open(fname+".pkl",'rb')
    data=pickle.load(pkl_file) 
    pkl_file.close()
    return data  

def generate_graph(distances,eps=20):
    return distances<eps



def compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))





def get_n_vertex(shapes):
    n_vertex=[]
    for s in shapes:
        n_vertex.append(s[0][1])
    return n_vertex

def get_out_vertex(shapes):
    n_vertex=[]
    for s in shapes:
        n_vertex.append(s[1][1])
    return n_vertex[0]

def get_in_fn(shapes):
    f_num=[]
    for s in shapes:
        f_num.append(s[0][2])
    return f_num

def get_out_fn(shapes):
    f_num=[]
    for s in shapes:
        f_num.append(s[1][2])
    return f_num

def get_input_dim(shapes):
    input_dim=0
    for s in shapes:
        input_dim+=s[0][1]*s[0][2]    
    return input_dim

def get_output_dim(shapes,n_pred,first_only=True):
    output_dim=0
    for s in shapes:
        output_dim+=s[1][1]*s[1][2]
        if first_only:
            break
    return output_dim*n_pred


def get_input_shape(shapes):
    input_shape=[]
    for s in shapes:
        input_shape.append((s[0][1],s[0][2]))
    return input_shape

def get_output_shape(shapes):
    output_shape=[]
    for s in shapes:
        output_shape.append((s[1][1],s[1][2]))
    return output_shape


def check_special_field(string, special_fields):
    for field in special_fields:
        if field in string:
            return True
    return False


def compress_ts(data,time_length=6,operator='mean'): #time length = sampling rate//5
    if operator=='sum':
        compressed_data = data.reshape((data.shape[0], -1, time_length,data.shape[2],data.shape[3])).sum(axis=2)
    else:
        compressed_data = data.reshape((data.shape[0], -1, time_length,data.shape[2],data.shape[3])).mean(axis=2)
    return compressed_data

