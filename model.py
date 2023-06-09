#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import scipy.io as scio
import numpy as np
import pywt
from scipy import signal
from matplotlib import pylab
from pylab import *
from sklearn.metrics import fowlkes_mallows_score
import itertools
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from coclust.evaluation.external import accuracy


# In[ ]:

''' Spike Detector Methods'''

def spikedet(signal, threshold):
    detection = []
    for id in range(len(signal)):
        if signal[id] > threshold:
            detection.append(id)
    return detection

factor = lambda i: 24/int(i)+1 if int(i)!=24 else 1
def neofilter(v,t):
    neo_value = [0 for _ in range(len(t))]
    peaks = []
    #neo calculation
    for i in range(1,len(t)-1):
        neo_value[i] = v[i]*v[i] - v[i-1]*v[i+1]
    neo_value[0] = neo_value[1]
    neo_value[-1] = neo_value[-2]
    return neo_value

def spikefilter(peaks,strip):
    x = np.array(peaks)
    len_old = x.shape[0]
    for i in range(len_old):
        if i >= x.shape[0] - 1:
            break
        for j in range(strip):#11
            if i >= x.shape[0] - 1:
                break
            if (x[i+1] - x[i]) < strip:#原来10，24k
            #if (x[i+1] - x[i]) < 6: #for DWT
                x = np.delete(x,i+1)
    return x

def dacalculation(groundtruth,detection):
    TP = FP = FN = 0
    misalignment = 0
    i = j = 0
    while((i<len(groundtruth)) & (j < len(detection))):
        if (detection[j] - groundtruth[i] <= 26) & ((detection[j] - groundtruth[i]) >= 0) : #detection succeeds, true positive, pointer both increases
            TP += 1
            misalignment += detection[j] - groundtruth[i]
            i += 1
            j +=1
        elif(detection[j] - groundtruth[i]) > 26: #a spike is missed, false negtive, only ground truth pointer increase
            FN +=1
            i += 1
        else: #noise is detected, false positive, pointer of detection increase
            FP += 1
            j += 1

    FN = FN + (len(groundtruth) - i)
    FP = FP + (len(detection) - j)
    miss = misalignment/TP
    DA = TP/(TP+FN+FP)
    
#     print("TP,FN,FP,DA:")
#     print(TP,FN,FP,DA)
    return DA,miss

def smoothing(datain,t):
    window = signal.windows.boxcar(5)
    smooth = [0 for _ in range(len(t))]
    for i in range(2,len(t)-2):
        smooth[i] = window[0] * datain[i-2] + window[1] * datain[i-1]+ window[2] * datain[i]+ window[3] * datain[i+1]+ window[4] * datain[i+2]
    smooth[0] = window[2] * datain[0] + window[3] * datain[1]+ window[4] * datain[2]
    smooth[1] = window[1] * datain[0] + window[2] * datain[1] + window[3] * datain[2]+ window[4] * datain[3]
    smooth[-1] = window[0] * datain[-3] + window[1] * datain[-2]+ window[2] * datain[-1]
    smooth[-2] = window[0] * datain[-4] + window[1] * datain[-3] + window[2] * datain[-2]+ window[3] * datain[-1]
    
    return np.array(smooth)


# In[ ]:


def sigma_calculation(v,sigma_a, G, T):
    #values = []
    num = 0
    ratio = 0
    #sigma_a = 0.5 #initial number, not sure the set value
    sigma_t = 0.3173
#     for t,y in v:
#         values.append(y)
#     std = np.std(values)
    for y in v:
        if abs(y) > sigma_a:
            num += 1
    ratio = num/len(v)
    error = ratio - sigma_t
    new_sigma_a = sigma_a + G * T * error
    return new_sigma_a


# In[ ]:
''' Spike Sorter methods'''

class Sorter:
    ''' Implementation of the algorithm'''
    def __init__(self,distance,threshold,update):
        ## Initialize the cluster templates
        self.distance = distance
        self.threshold= threshold
        self.cluster_means=[]
        self.cluster_cardinality = []
        self.update=update
    
    def closest_cluster(self,signal,test=False):
        # Compute distances to all centroids and return closest cluster if closer than threshold
        ## Otherwise create a new cluster
        min_dist= float("inf")
        for i,center in enumerate(self.cluster_means):
            if(self.cluster_cardinality[i]==-1):
                continue
            if(test and self.cluster_cardinality[i]<np.sum(self.cluster_cardinality)*0.05):
                continue
            dist = self.distance(signal,center)
            if(dist<min_dist):
                min_dist=dist
                ind= i
        if(test):
            try:
                return ind
            except:
                return -1
        if(min_dist<self.threshold):
            return ind
        else:
            return len(self.cluster_means)
        
    def train(self,signal,test=False):
        ## Assign the signal to a cluster and update the cluster centroid
        ## The update follows the update function, which is a parameter of the model if the cluster is an old cluster
        ## If the cluster is new the signal is considered as its centroid
        if(len(self.cluster_means)==0):
            self.cluster_means.append(signal)
            self.cluster_cardinality.append(1)
            return 0
        cluster = self.closest_cluster(signal,test)
        if(cluster==len(self.cluster_means)):
            self.cluster_means.append(signal)
            self.cluster_cardinality.append(1)
            return cluster
        self.cluster_cardinality[cluster]=self.cluster_cardinality[cluster]+1
        self.cluster_means[cluster] =self.update(self.cluster_cardinality[cluster],self.cluster_means[cluster],signal)       
                

        return cluster
    def get_centroids(self):
        return self.cluster_means
        
                
        


# In[ ]:


def thresh_calc(sigma,i):
    ## Formula to compute the threshold based on the variance of the signal
    return (-1/64 + 13 * sigma/32)*factor(i)


# In[ ]:


def feature_extraction(signal,fac):
    ## Function for feature selection
    fd = np.diff(signal)
    sd = np.diff(fd)
    rankfd = np.argsort(fd)
    ranksd=np.argsort(sd)
    return np.array([min(fd),max(fd),min(sd),max(sd),np.argmax(signal)*3*fac/64])


# In[ ]:


def l1(x,y):
    ## L1 Norm
    return  np.linalg.norm(x-y,ord=1)
def l2(x,y):
    ## L2 Norm Squared
    return  np.linalg.norm(x-y,ord=2)**2


# In[ ]:


def hypertune(params,datas,gts,update,merging=False):
    ## Returns the best parameters for the distance metric and threshold based on the classification accuracy on the provided data
    keys, values = zip(*params.items())
    permutations_dict = [dict(zip(keys, v)) for v in itertools.product(*values)]
    max_score = 0
    for param in tqdm(permutations_dict):
        score = 0
        for i in range(len(datas)):
            data = datas[i]
            gt= gts[i]
            X_train, X_test, y_train, y_test = train_test_split(data, gt, test_size=0.2,shuffle=True)
            sorter = Sorter(param['distance'],param['thresh'],update)
            c_hat = np.array([sorter.train(x) for x in X_train])
            c_hat = np.array([sorter.train(x,test=True) for x in X_test])
            score += accuracy(y_test,c_hat)
        
        if score>max_score and (-1) not in set(c_hat):
            best_sorter = sorter
            max_score = score
            best_param = param
            best_pred= c_hat
            
    rng = range(len(X_test[0]))
    colors= ['r','b','g','y','w','k','c','m']
    color_dict={}
    i=0
    for v in list(set(best_pred)):
        color_dict[v] = colors[i]
        i+=1
    plt.figure()
    fig,axs = plt.subplots(1,2, figsize=(15, 5))
    for i in range(len(X_test)):
        axs[0].plot(rng,X_test[i],c= color_dict[best_pred[i]],alpha=.1)
        axs[0].set_title('predictions')
    for i in range(len(X_test)):
        axs[1].plot(rng,X_test[i],c='rgbk'[y_test[i]-1],alpha=.1)
        axs[1].set_title('groud truth')
    plt.show()
    print(best_param,max_score/len(datas))
    return best_param,max_score/len(datas)


# In[ ]:

exact_update = lambda card,mean,signal:  ((card-1) * mean + signal)/card
approx_update = lambda card,mean,signal: mean + (-mean + signal)/2**(np.ceil(max(np.log2(card+1),256)))
def train_predict(param,data,gt,update,merging=False):
    ## applies the model on the provide data
    X_train, X_test, y_train, y_test = train_test_split(data, gt, test_size=0.2,shuffle=True)
    sorter = Sorter(param['distance'],param['thresh'],update)
    c_hat = np.array([sorter.train(x) for x in X_train])
    c_hat = np.array([sorter.train(x,test=True) for x in X_test])
    score = accuracy(y_test,c_hat)
    
    return score

