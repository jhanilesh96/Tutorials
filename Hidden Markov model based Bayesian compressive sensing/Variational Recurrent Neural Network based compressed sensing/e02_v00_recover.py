'''
Compressive sensing deals with problems where recovery is done with less observations than the number of variables. 
In such a case, a nprior is an assumption over the model that helps in recovery.
We will work with complex variables as much as possible


'''


import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import scipy 

import sys, os
sys.path.insert(0, '..')
import utils as myutils

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



from e00_v00_genData import GenData
genData = GenData()
import e01_v00_vrnn as vrnn
vrnn.savefigs()

import tensorflow as tf
import tensorflow_probability as tfp





Phi_prior = vrnn.Phi_prior;
dim_g = vrnn.dim_g;
dim_z = vrnn.dim_z;
dim_h = vrnn.dim_h;
Phi_dec_mean = vrnn.Phi_dec_mean;
Phi_enc_mean = vrnn.Phi_enc_mean;
Phi_h = vrnn.Phi_h;
f_theta = vrnn.f_theta;


for mlp in vrnn.mlps:
    mlp.trainable=False


H = genData.sample()


TRIALS = H.shape[0]
T = H.shape[1]
N = H.shape[2]*H.shape[3]
K = N//2
A = np.random.randn(*[K,N]).astype(ftype)
VAR_N = 1e-3



'''
GA MMSE baseline
'''
def estimateGAMMSE(h_true_t, y, A, VAR_N):
    fH = lambda _X : np.conjugate(np.transpose(_X))
    eye = lambda _M : np.eye(*[_M,_M], dtype=ftype)
    inv = np.linalg.inv
    s_idx = h_true_t>VAR_N;
    h_mmse_t = np.zeros(h_true_t.shape, h_true_t.dtype)
    h_mmse_t = np.zeros(N, dtype=ctype)
    h_mmse_t[s_idx] = inv(fH(A[:,s_idx])@A[:,s_idx])@fH(A[:,s_idx])@y
    return h_mmse_t


'''
Lasso baseline
'''
def estimateLasso(y, A, VAR_N):
    fH = lambda _X : np.conjugate(np.transpose(_X))
    eye = lambda _M : np.eye(*[_M,_M], dtype=ftype)
    inv = np.linalg.inv
    K, N = A.shape
    h = cp.Variable(N)
    _obj = (1e-3/VAR_N)*cp.norm(y-A@h,2)**2 + 0.1*cp.norm(h, 1)
    _ = cp.Problem(cp.Minimize(_obj), []).solve()
    return h.value



def postProcess(h_pred_t, y, A, VAR_N):
    fH = lambda _X : np.conjugate(np.transpose(_X))
    eye = lambda _M : np.eye(*[_M,_M], dtype=ftype)
    inv = np.linalg.inv
    s_idx = h_pred_t[0,:,:,0].numpy().ravel()>VAR_N;
    if sum(s_idx)>np.size(y):
        # in case a prediction is severely wrong, should not occur frequently
        print('##', end='', flush=True)
        s_idx = np.argsort(h_pred_t[0,:,:,0].numpy().ravel())[-np.size(y)//3:]
    h_post_t = np.zeros(h_true_t.shape, h_true_t.dtype)
    h_post_t = np.zeros(N, dtype=ctype)
    h_post_t[s_idx] = inv(fH(A[:,s_idx])@A[:,s_idx])@fH(A[:,s_idx])@y
    # nmse(h_true_t.reshape(H.shape[2:]), h_post_t.reshape(H.shape[2:])) 
    return h_post_t

'''
At each time step, we solve for 
    argmin_{z}  |y-AG(z)|^2/VAR_N + logp G(z|g_t_1)
Or
    argmin_{z,h} 
'''
def optimizeForZ(g_t_1, y, A, VAR_N=VAR_N, NUM_STEPS=50, lr=1e-5):
    mean_z_t_prior, logvar_z_t_prior = Phi_prior(g_t_1)
    zo_t = tf.Variable(mean_z_t_prior)
    nll = lambda: ((2*VAR_N)**-1)*tf.reduce_sum(tf.square(y - tf.linalg.matvec(A,tf.reshape(Phi_dec_mean(tf.concat([zo_t,g_t_1], axis=-1)),[N]))));
    nprior = lambda: 0.5*tf.reduce_sum(logvar_z_t_prior + tf.divide(tf.square(zo_t - mean_z_t_prior),tf.exp(logvar_z_t_prior)))
    loss_fn = lambda:nll()+nprior()
    train_opt = tf.keras.optimizers.SGD(learning_rate=lr)
    for __ in range(NUM_STEPS):    
        with tf.GradientTape() as tape:
            loss = loss_fn()
        gradients = tape.gradient(loss, [zo_t])
        train_opt.apply_gradients(zip(gradients, [zo_t]))
    return zo_t, Phi_dec_mean(tf.concat([zo_t,g_t_1], axis=-1)), nll(), nprior()
    # log p()


n_idx=1; t_idx=0
g_t_1 = g_0 = 1e-2*tf.ones(shape=[1,dim_g]);
h_true_t_list = []
h_mmse_t_list = []
h_pred_t_list = []
nll_list = []
nprior_list = []
while t_idx<T:
    print(str(t_idx)+'...',end='',flush=True)
    h_true_t = H[n_idx, t_idx].ravel()
    # assert np.allclose(h_true.reshape(H.shape[2:]), H[n_idx, t_idx])
    # Get Observation
    y_noiseless = A@h_true_t
    y = (y_noiseless + np.sqrt(VAR_N)*np.random.randn(K)).astype(ftype)
    # Recover a good Approximation
    # NOT USED:: but if at any time index we want to reset search
    if t_idx==-1:
        h_pred_t =  H[n_idx, t_idx][np.newaxis, :,:,np.newaxis]
        ho_t = Phi_h(h_pred_t[:,:,:,tf.newaxis])
        zo_t = Phi_enc_mean(tf.concat([ho_t,g_t_1],axis=-1))
    else:
        zo_t, h_pred_t, nll, nprior = optimizeForZ(g_t_1, y, A, VAR_N, NUM_STEPS=100 if t_idx==0 else (50 if t_idx==1 else 20))    
        ho_t = Phi_h(h_pred_t[:,:,:,tf.newaxis])
    # Update Digest vector based on Approximation
    g_t = f_theta(tf.concat([ho_t,zo_t,g_t_1], axis=-1))
    g_t_1 = g_t
    h_true_t_list.append(h_true_t.reshape(h_pred_t.shape))
    h_pred_t_list.append(h_pred_t)
    # mmse estimate
    h_mmse_t = estimateGAMMSE(h_true_t, y, A, VAR_N)
    h_mmse_t_list.append(h_mmse_t)
    # nll_list.append(nll)
    # nprior_list.append(nprior)2
    # post processing esimtation
    h_post_t = postProcess(h_pred_t, y, A, VAR_N)
    h_lass_t = estimateLasso(y, A, VAR_N)
    print(\
        'Pred:', cfp(nmse(h_true_t.reshape(H.shape[2:]), h_pred_t[0,:,:,0]), p=6), \
            ', PostP:', cfp(nmse(h_true_t.reshape(H.shape[2:]), h_post_t.reshape(H.shape[2:])), p=6),\
                ', Las:', cfp(nmse(h_true_t.reshape(H.shape[2:]), h_lass_t.reshape(H.shape[2:])), p=6),\
                    ', GA:', cfp(nmse(h_true_t.reshape(H.shape[2:]), h_mmse_t.reshape(H.shape[2:])), p=6),\
            )
    t_idx = t_idx+1


print()
h_true = np.array(h_true_t_list)[:,0,:,:,0]
h_pred = np.array(h_pred_t_list)[:,0,:,:,0]


_ = plt.subplots(nrows=h_true.shape[0], ncols=2)
for t_idx in range(h_true.shape[0]):
    _ = plt.subplot(h_true.shape[0], 2, 2*t_idx + 1)
    _ = plt.xlabel('true, t='+str(t_idx))
    _ = plt.imshow(h_true_t_list[t_idx].reshape(H.shape[2:]))
    _ = plt.subplot(h_true.shape[0], 2, 2*t_idx + 2)
    _ = plt.xlabel('pred, t='+str(t_idx))
    _ = plt.imshow(h_pred[t_idx])

plt.show()



## DEBUG::
# exit()





# Phi_dec_mean.trainable=False
# H = genData.sample()

# TRIALS = H.shape[0]
# T = H.shape[1]
# N = H.shape[2]*H.shape[3]
# K = N//3
# A = np.random.randn(*[K,N]).astype(ftype)
# VAR_N = 1e-2


# n_idx=0; t_idx=0
# g_t_1 = g_0 = 1e-2*tf.ones(shape=[1,dim_g]);
# h_true_t_list = []
# h_pred_t_list = []
# nll_list = []
# nprior_list = []
# h_true_t = H[n_idx, t_idx].ravel()
# # assert np.allclose(h_true.reshape(H.shape[2:]), H[n_idx, t_idx])
# # Get Observation
# y_noiseless = A@h_true_t
# y = (y_noiseless + np.sqrt(VAR_N)*np.random.randn(K)).astype(ftype)
# # Recover a good Approximation

# mean_z_t_prior, logvar_z_t_prior = Phi_prior(g_t_1)
# zo_t = tf.Variable(mean_z_t_prior)
# y_hat = tf.linalg.matvec(A,tf.reshape(h_pred,[N]))
# nll = lambda: ((2*VAR_N)**-1)*tf.reduce_sum(tf.square(y - y_hat));
# nprior = lambda: 0.5*tf.reduce_sum(logvar_z_t_prior + tf.divide(tf.square(zo_t - mean_z_t_prior),tf.exp(logvar_z_t_prior)))
# loss_fn = lambda:nll()+nprior()
# train_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

# with tf.GradientTape() as tape:
#     h_pred = Phi_dec_mean(tf.concat([zo_t,g_t_1],axis=-1))
#     # loss = nll()
#     loss = h_pred

# gradients = tape.gradient(loss, zo_t)
# train_opt.apply_gradients()

