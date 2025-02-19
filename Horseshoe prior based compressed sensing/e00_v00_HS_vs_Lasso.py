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

np.random.seed(3254)

_CN = myutils._CN
nmse = myutils.nmse
ctype = myutils.ctype
ftype = myutils.ftype
markov = myutils.markov1D
woodburyInv = myutils.woodburyInv
debug = 2;
cfp = myutils.convertFloattoPrec

mvn = scipy.stats.multivariate_normal


VAR_N = 1e-3
K = 50
N = 100
p = 0.1
A = np.random.randn(*[K, N])
s = np.random.rand(N)<p
h_GLOBAL = np.random.rand(N)*s

def genData(VAR_N=VAR_N):
    y = A@h_GLOBAL + np.sqrt(VAR_N)*np.random.randn(K)
    return y, h_GLOBAL



def estimateLasso(y, A, VAR_N, lam=0.1):
    fH = lambda _X : np.conjugate(np.transpose(_X))
    eye = lambda _M : np.eye(*[_M,_M], dtype=ftype)
    inv = np.linalg.inv
    K, N = A.shape
    _h = cp.Variable(N)
    _obj = cp.norm(y-A@_h,2)**2 + lam*cp.norm(_h, 1)
    _ = cp.Problem(cp.Minimize(_obj), []).solve()
    return _h.value




# Mean field variational Bayes for continuous sparse signal shrinkage: Pitfalls and remedies
# Model III
def estimateHorseShoe_III(y, A, VAR_N, vec_mu_q_c=0.1, C=1e-3, iters=100, debug=False, zeroInit=True, h_init=None):
    fH = lambda _X : np.conjugate(np.transpose(_X))
    eye = lambda _M : np.eye(*[_M,_M], dtype=ftype)
    inv = np.linalg.inv
    K, N = A.shape
    if h_init is None:
        if zeroInit:
            h_t_m1 = 1e-3+np.zeros(N)
            h_t = 1e-3+np.zeros(N)
        else:
            h_t_m1 = h_t = (np.linalg.inv((np.conj(A.T)@A))@np.conj(A.T)@y)
    else:
        h_t_m1 = h_t = h_init
    vec_mu_q_c = vec_mu_q_c * np.ones(h_t.shape)
    mu_q_sigma2_inv = 1.0;
    for i in range(iters):
        mu_q_a_inv = C**2 / (C**2 * mu_q_sigma2_inv + 1)
        vec_G = 0.5 * mu_q_sigma2_inv * np.abs(h_t)**2
        vec_mu_q_b = 1/(vec_G + vec_mu_q_c)
        vec_mu_q_c = 1/(vec_mu_q_b+1)
        h_var = 1/(mu_q_sigma2_inv*vec_mu_q_b)
        c1 = 1/VAR_N; c2 = 0; C3 = 1/h_var
        Sigma = np.linalg.inv(c1*(np.conj(A.T)@A) + C3*np.eye(h_t.size))
        Mu = Sigma@(c1*np.conj(A.T)@y)
        h_t_m1 = h_t
        h_t = Mu
        mu_q_sigma2_inv = (h_t.shape[0]+1)/(2*mu_q_a_inv + np.sum(vec_mu_q_b * np.abs(h_t)**2))
        if np.any(np.isnan(h_t)) or np.linalg.norm(h_t_m1-h_t)<1e-4:
            return h_t_m1
        if debug:
            print(i, nmse(h, h_t))
    return h_t


_alg = estimateHorseShoe_III
# _log_VAR_N=-5
# y, h = genData(10.0**_log_VAR_N)
# nmse(h, _alg(y, A, VAR_N=10.0**_log_VAR_N, debug=True))

y, h = genData()
print('Lasso::',nmse(h, estimateLasso(y, A, VAR_N)))
print('Horse::',nmse(h, _alg(y, A, VAR_N)))
# for loglam in np.arange(-10,5):
#     print(loglam, nmse(h, estimateLasso(y, A, VAR_N,lam=10.0**loglam)))


logVAR_Ns = np.arange(-5,-1,0.5)
logLams = np.arange(-10,5,0.5)
VAL_TRIALS = 20;
def CrossValidation():
    lam_cv = {}
    # _log_VAR_N = logVAR_Ns[0]
    for _log_VAR_N in logVAR_Ns:
        nmses = np.zeros([len(logLams), VAL_TRIALS])
        for _loglam_idx, _loglam in enumerate(logLams):
            for _trial_idx in range(VAL_TRIALS):
                y, h = genData(10.0**_log_VAR_N)
                nmses[_loglam_idx, _trial_idx] = nmse(h, estimateLasso(y, A, VAR_N=10.0**_log_VAR_N,lam=10.0**_loglam))
        min_argidx = np.argmin(np.mean(nmses, axis=-1))
        lam_cv[_log_VAR_N] = logLams[min_argidx]
    return lam_cv


TEST_TRIALS = VAL_TRIALS*5
lam_cv = CrossValidation(); print(lam_cv)
nmses_lasso = np.zeros([len(logVAR_Ns), TEST_TRIALS])
nmses_horse = np.zeros([len(logVAR_Ns), TEST_TRIALS])
for _log_VAR_N_idx, _log_VAR_N in enumerate(logVAR_Ns):
    for _trial_idx in range(TEST_TRIALS):
        y, h = genData(10.0**_log_VAR_N)
        h_lasso = estimateLasso(y, A, VAR_N=10.0**_log_VAR_N,lam=10.0**lam_cv[_log_VAR_N])
        nmses_lasso[_log_VAR_N_idx, _trial_idx] = nmse(h, h_lasso)
        h_horse = _alg(y, A, VAR_N=10.0**_log_VAR_N, h_init=None)
        nmses_horse[_log_VAR_N_idx, _trial_idx] = nmse(h, h_horse)
    print(_log_VAR_N, cfp(np.mean(nmses_lasso[_log_VAR_N_idx]), p=5), cfp(np.mean(nmses_horse[_log_VAR_N_idx]), p=5), sep='\t')



plt.semilogy(logVAR_Ns, np.mean(nmses_lasso, axis=-1), label='lasso')
plt.semilogy(logVAR_Ns, np.mean(nmses_horse, axis=-1), label='horse')
plt.legend()
plt.grid(True)
plt.title('N=100, K=50, p=0.1')
plt.xlabel('Noise Variance')
plt.ylabel('log NMSE')
plt.show()








'''
In this tutorial, we design a horseshoe prior based sparse recovery. This prior provides better shrinkage and an i.i.d. support 

'''

# Mean field variational Bayes for continuous sparse signal shrinkage: Pitfalls and remedies
# Model III
def estimatePareto_III(y, A, VAR_N, vec_mu_q_c = 0.1, C = 1e-3, lam=5, iters=100, debug=False, zeroInit=True, h_init=None):
    fH = lambda _X : np.conjugate(np.transpose(_X))
    eye = lambda _M : np.eye(*[_M,_M], dtype=ftype)
    inv = np.linalg.inv
    K, N = A.shape
    if h_init is None:
        if zeroInit:
            h_t_m1 = 1e-3+np.zeros(N)
            h_t = 1e-3+np.zeros(N)
        else:
            h_t_m1 = h_t = (np.linalg.inv((np.conj(A.T)@A))@np.conj(A.T)@y)
    else:
        h_t_m1 = h_t = h_init
    vec_mu_q_c = vec_mu_q_c * np.ones(h_t.shape)
    for i in range(iters):
        mu_q_a_inv = C**2 / (C**2 * (1/VAR_N) + 1)
        vec_G = 0.5 * (1/VAR_N) * np.abs(h_t)**2
        vec_mu_q_b = np.sqrt(vec_mu_q_c/vec_G)
        vec_mu_q_b_inv = 1/vec_mu_q_b + 1/(2*vec_mu_q_c)
        vec_mu_q_c = (lam+1)/(vec_mu_q_b_inv+1)
        h_var = 1/((1/VAR_N)*vec_mu_q_b)
        c1 = 1/VAR_N; c2 = 0; C3 = 1/h_var
        Sigma = np.linalg.inv(c1*(np.conj(A.T)@A) + C3*np.eye(h_t.size))
        Mu = Sigma@(c1*np.conj(A.T)@y)
        h_t_m1 = h_t
        h_t = Mu
        if np.any(np.isnan(h_t)) or np.linalg.norm(h_t_m1-h_t)<1e-4:
            return h_t_m1
        if debug:
            print(i, nmse(h, h_t))
    return h_t





def estimatePareto_II(y, A, VAR_N, vec_mu_q_c = 0.1, C = 1e-3, lam=5, iters=100, debug=False, zeroInit=True, h_init=None):
    fH = lambda _X : np.conjugate(np.transpose(_X))
    eye = lambda _M : np.eye(*[_M,_M], dtype=ftype)
    inv = np.linalg.inv
    K, N = A.shape
    if h_init is None:
        if zeroInit:
            h_t_m1 = 1e-3+np.zeros(N)
            h_t = 1e-3+np.zeros(N)
        else:
            h_t_m1 = h_t = (np.linalg.inv((np.conj(A.T)@A))@np.conj(A.T)@y)
    else:
        h_t_m1 = h_t = h_init
    vec_mu_q_c = vec_mu_q_c * np.ones(h_t.shape)
    mu_q_sigma2_inv = 1.0;
    for i in range(iters):
        mu_q_a_inv = C**2 / (C**2 * mu_q_sigma2_inv + 1)
        vec_G = 0.5 * mu_q_sigma2_inv * np.abs(h_t)**2
        vec_mu_q_b = np.sqrt(vec_mu_q_c/vec_G)
        vec_mu_q_b_inv = 1/vec_mu_q_b + 1/(2*vec_mu_q_c)
        vec_mu_q_c = (lam+1)/(vec_mu_q_b_inv+1)
        h_var = 1/(mu_q_sigma2_inv*vec_mu_q_b)
        c1 = 1/VAR_N; c2 = 0; C3 = 1/h_var
        Sigma = np.linalg.inv(c1*(np.conj(A.T)@A) + C3*np.eye(h_t.size))
        Mu = Sigma@(c1*np.conj(A.T)@y)
        h_t_m1 = h_t
        h_t = Mu
        mu_q_sigma2_inv = (h_t.shape[0]+1)/(2*mu_q_a_inv + np.sum(vec_mu_q_b * np.abs(h_t)**2))
        if np.any(np.isnan(h_t)) or np.linalg.norm(h_t_m1-h_t)<1e-4:
            return h_t_m1
        if debug:
            print(i, nmse(h, h_t))
    return h_t
