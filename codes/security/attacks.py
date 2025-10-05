import utils
import scripts.loss_ as loss_
import torch
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans

def l2loss(x):
    return (x**2).mean()

def get_source_embeddings(model,loader,args,structure_type):
    adj=args.adj
    n_pred=args.n_pred
    N=utils.get_out_vertex(args.raw_data_shapes)
    model.eval()
    raw_data=[]
    source_embeddings=[]
    for unit in loader:   
        data,target=unit[0],unit[1]
        B=data.shape[0]
        target=target.reshape(B,n_pred,N,-1)
        
        if structure_type=='HSTFL':
            source_data,source_res=model.GetSourceOutput(A=adj,x=data)
        else:
            source_data,source_res=model.GetTemporalOutput(A=adj,x=data)
        raw_data.append(source_data)
        source_embeddings.append(source_res)

    return raw_data,source_embeddings


def inverse_data_samples(model,source_embeddings,args,target_size,structure_type,data_indexs=(4,3)):
    #step1 select data to inverse
    target_size=[1,target_size[0],target_size[1],target_size[2]]
    xGen = torch.zeros(target_size, requires_grad = True, device="cuda")
    print('start attacking at ',data_indexs)
    optimizer = optim.Adam(params = [xGen], lr = 0.1, amsgrad = True)
    embeddings=[]
    if structure_type=='HSTFL':
        "ATTACK on HSTFL"
        for i in range(len(source_embeddings[0])):
            tmp=source_embeddings[data_indexs[0]][i][data_indexs[1]]
            embeddings.append(tmp)
        embeddings=torch.stack(embeddings)
        L,N,F=embeddings.shape
        embeddings=embeddings.reshape(L,1,N,F)
    else:
        "ATTACK on HSTFL-NoVNA"
        embeddings=source_embeddings[data_indexs[0]][data_indexs[1]]
        N,F=embeddings.shape
        embeddings=embeddings.reshape(1,N,F)    
    #step 2 Optimizer
    NIters=500
    lambda_l2=1e-4
    lambda_l1=1e-5#1e-3

    #step 3 inverse data through GD    
    adj=args.adj
    bestGen=xGen
    best_featureLoss=10000
    for i in range(NIters):
        optimizer.zero_grad()
        if structure_type=='HSTFL':
            _,xFeature = model.GetSourceOutput(A=adj,x=xGen,need_preprocess=False)
        else:
            _,xFeature = model.GetTemporalOutput(A=adj,x=xGen,need_preprocess=False)
        featureLoss = ((xFeature - embeddings)**2).mean()

        normLoss = l2loss(xGen)
        smoothness=loss_.smoothness_loss(xGen)
        totalLoss = featureLoss + lambda_l1*smoothness + lambda_l2 * normLoss

        totalLoss.backward(retain_graph=True)
        optimizer.step()
        if best_featureLoss>featureLoss:
            best_featureLoss=featureLoss
            bestGen=xGen.clone()

        if (i+1)%50==0:
            print ("Iter ", i+1, "Feature loss: ", featureLoss.cpu().detach().numpy(),  "smoothness Loss: ", smoothness.cpu().detach().numpy(), "l2Loss: ", normLoss.cpu().detach().numpy())




    inversed_data=bestGen
    return inversed_data,data_indexs

def eval_attacks(target_data,reconstructed_data,data_indexs=[5,3],scaler=None,id=(4,3),fn=''):
    print('Evaluating attack performance')
    metric=[loss_.masked_mae,loss_.masked_rmse,loss_.masked_smape,loss_.DTW]
    target=target_data[data_indexs[0]][data_indexs[1]]
    target=target.reshape(1,target.shape[0],target.shape[1],target.shape[2])
    pred=reconstructed_data#[data_indexs[0]][data_indexs[1]]
    vector1=torch.full_like(pred, 0) #mean approximation
    vector2=torch.rand_like(pred) #uniform random approximation
    vector3=torch.rand_like(pred) #Guassian random approximation
    # print(target.shape,pred.shape)

    metric_kind=0
    print("Sample ID",id)
    all_result=[]
    for metric_kind in [0,1]:
        result=torch.tensor([metric[metric_kind](target,pred),metric[metric_kind](target,vector1),metric[metric_kind](target,vector2),metric[metric_kind](target,vector3)])
        print('Attack result',metric[metric_kind](target,pred))
        print('mean approximate',metric[metric_kind](target,vector1))
        print('uniform approximate',metric[metric_kind](target,vector2))
        print('Gaussian approximate',metric[metric_kind](target,vector3))
        all_result.append(result)
    save=True
    if save:
        attack_result={'pred':pred.detach().cpu().numpy(),'target':target.detach().cpu().numpy()}
        np.savez('security/attack_result/attack_result_'+fn+str(id)+'.npz',**attack_result)
    return torch.stack(all_result)

def sample_time_series(time_series):
    bs=time_series[0].shape[0]
    time_series=torch.cat(time_series)
    ids=kmeans_clustering(time_series)
    result=[]
    for id in ids:
        result.append((id//bs,id%bs))

    print('Sampled ID')
    print(result)
    return result

def kmeans_clustering(timeseries, num_clusters=10):
    reshaped_series = timeseries.view(timeseries.shape[0],-1).cpu().numpy()
    #print(reshaped_series.shape)
    #1/0

    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(reshaped_series)

    cluster_centers =kmeans.cluster_centers_
    #print(cluster_centers.shape)
    labels = torch.from_numpy(kmeans.labels_)
    representative_ids = []
    #print(cluster_centers.device,timeseries.device)
    for i in range(num_clusters):
        cluster_indices = torch.where(labels == i)[0]
        #print(cluster_indices)
        v=reshaped_series[cluster_indices ] - cluster_centers[i]
        distances = torch.norm(torch.tensor(v), dim=1)
        #print(distances.shape)
        closest_id=torch.argmin(distances)
        #print(closest_id)
        closest_index = cluster_indices[closest_id]
        representative_ids.append(closest_index.item())

    print('representative_ids',representative_ids)
    return representative_ids