'''
Compressive sensing deals with problems where recovery is done with less observations than the number of variables. 
In such a case, a prior is an assumption over the model that helps in recovery.
We will work with complex variables as much as possible
'''


import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import scipy 

import sys, os
sys.path.insert(0, '..')
import utils as myutils
# Let us assume an average sparsity of 10% with N=300 observations and K=600 variables

# np.random.seed(87346)

_CN = myutils._CN
nmse = myutils.nmse
ctype = myutils.ctype
ftype = myutils.ftype
markov = myutils.markov1D
woodburyInv = myutils.woodburyInv
debug = 2;
cfp = myutils.convertFloattoPrec

mvn = scipy.stats.multivariate_normal

#%% System Parameters


class Empty():
    def __init__(self):
        pass;

self = Empty()


class GenData():
    def __init__(self, K=[25,25], T=15, grid_fine=1, base_speed=[0.25, 0.25], base_variance=0.1):
        # System parameters
        self.Kx, self.Ky = K
        self.T = T;
        self.grid_fine = grid_fine
        self.base_speed_min, self.base_speed_max = base_speed
        self.base_variance = 0.2;
        self.setPalleteParameters()
        self.last_sample = None
    def setPalleteParameters(self, xm=[-4,4], ym=[-4,4]):
        self.x_min, self.x_max = xm;
        self.y_min, self.y_max = ym;
        self.x, self.y = np.meshgrid(np.linspace(self.x_min, self.x_max, self.Kx*self.grid_fine), np.linspace(self.y_min, self.y_max, self.Ky*self.grid_fine))
        self.xy = np.dstack([self.x,self.y])
    def sample(self, numSamples=5):
        H = np.zeros([numSamples, self.T, self.grid_fine*self.Kx, self.grid_fine*self.Ky], dtype=ftype);
        ############################################
        # choose number of clusters from poisson distribution truncated at 4
        # here we just use uniform [1,2]
        for sample in range(numSamples):
            num_clusters = np.random.randint(2)+1
            mean_clusters = np.array([np.array([0.75*(np.random.uniform()*(self.x_max-self.x_min)+self.x_min),\
                0.75*(np.random.uniform()*(self.y_max-self.y_min)+self.y_min)]) for __ in range(num_clusters)])
            vel_clusters = np.array([np.array([np.random.rand()*(self.base_speed_max-self.base_speed_min) + self.base_speed_min,\
                np.random.rand()*(self.base_speed_max-self.base_speed_min) + self.base_speed_min]) for __ in range(num_clusters)])
            cov_cluster_base_from = np.zeros([num_clusters, 2, 2])
            cov_cluster_base_to = np.zeros([num_clusters, 2, 2])
            for n_idx in range(num_clusters):
                temp = np.random.randn(*[2,2])
                cov_cluster_base_from[n_idx, :, :] = self.base_variance*(temp@temp.T + 0.1*np.eye(2))
                temp = np.random.randn(*[2,2])
                cov_cluster_base_to[n_idx, :, :] = self.base_variance*(temp@temp.T + 0.1*np.eye(2))
            for t_idx in range(self.T):
                for n_idx in range(num_clusters):
                    # each cluster starts at the mean_clusters and runs for T time steps with vel_clusters
                    _m = mean_clusters[n_idx] + vel_clusters[n_idx]*t_idx
                    # # it wraps at the edge
                    # _m[0] = (_m[0] - self.x_min)%(self.x_max-self.x_min) + self.x_min
                    # _m[1] = (_m[1] - self.y_min)%(self.y_max-self.y_min) + self.y_min
                    # it sticks to the edge
                    _m[0] = np.clip(_m[0], a_min=self.x_min, a_max=self.x_max)
                    _m[1] = np.clip(_m[1], a_min=self.y_min, a_max=self.y_max)
                    # cluster covariance evolves from _from to _to
                    _cov = (1-(t_idx/self.T))*cov_cluster_base_from[n_idx] + (t_idx/self.T)*cov_cluster_base_to[n_idx]
                    mvn_obj = mvn(mean=_m, cov=_cov)
                    H[sample, t_idx] += mvn_obj.pdf(self.xy)
        self.last_sample = H
        return self.last_sample



if __name__=='__main__':
    genData = GenData()
    H = genData.sample();
    _ = plt.subplots(nrows=H.shape[0], ncols=H.shape[1])
    for s_idx in range(H.shape[0]):
        for t_idx in range(H.shape[1]):
            _ = plt.subplot(H.shape[0],H.shape[1],s_idx*H.shape[1] + t_idx + 1)
            _ = plt.xlabel('s='+str(s_idx)+', t='+str(t_idx))
            _ = plt.imshow(H[s_idx][t_idx])
    plt.show()
    # for i in range(50):
    #     print('saving...',i)
    #     H = genData.sample(numSamples=200)
    #     myutils.saveobj(H, f='H_'+str(i), folderName='results_big')