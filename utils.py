
import numpy as np;
import cvxpy as cp
import scipy
import h5py  


def is_prime(x):
    for i in range(2, int(np.sqrt(x))+1):
        if x % i == 0:
            return False;
    return True

def next_prime(x):
    for a in range(x+1, 10*x):
        if is_prime(a):
            return a
    return None


ftype = 'float64'; 
ctype = 'complex'+str(2*int(ftype[-2:]));
c = lambda _, tf=None : tf.dtypes.cast(_, ctype)
f = lambda _, tf=None : tf.dtypes.cast(_, ftype)
CN = lambda _M : (1/np.sqrt(2)) * (np.random.randn(_M) + 1j*np.random.randn(_M))
_CN = lambda _M : (1/np.sqrt(2)) * (np.random.randn(*_M) + 1j*np.random.randn(*_M))
dot = lambda t=None : print('.'if t is None else t, end='', flush=True);
nfH = lambda _X : np.conjugate(np.transpose(_X))
nfC = lambda _X : np.conjugate(_X)
rr = lambda _X : _X.reshape([_X.shape[0], -1])
_nmse = lambda _X, _XHat : np.mean(np.linalg.norm(rr(_X) - rr(_XHat), axis=-1)**2)/np.mean(np.linalg.norm(rr(_X), axis=-1)**2)
nmse = lambda _X, _XHat : _nmse(_X.numpy() if 'tens' in str(type(_X)) else _X, _XHat.numpy() if 'tens' in str(type(_XHat)) else _XHat)



def markov1D(n=100, p01=0.12, p10=0.7):
    m = np.zeros(n)
    m[0] = np.random.rand()<(p01/(p10+p01))
    for i in range(n-1):
        if m[i]==0:
            m[i+1] = 1 if np.random.rand()<p01 else 0    
        else:
            m[i+1] = 0 if np.random.rand()<p10 else 1
    return m;


'''
Short : np.allclose(woodburyInv(D, A), inv(fH(A)@A + np.diag(D)))
Usage : invert large covariance matrix of form inv(D + A.H @ A)
Input : D is diagonal of NxN matrix, shape = N
        A shape = K, N
        if V and U are None, they are determined from A
Output : inv(D + A.H @ A) or inv(D + U @ V)

'''

def woodburyInv(D, A, V=None, U=None, assertion=False):
    fH = lambda _X : np.conjugate(np.transpose(_X))
    eye = lambda _M, dtype='complex128' : np.eye(*[_M,_M], dtype=dtype)
    inv = np.linalg.inv
    D_inv_vec = 1/D
    U = fH(A) if U is None else U
    V = A if V is None else V
    I = eye(A.shape[0], dtype=A.dtype) # K - rank
    Wood_inv = np.diag(D_inv_vec) - D_inv_vec[:,np.newaxis] * (U @ inv(I + (V*D_inv_vec)@U) @ V) * D_inv_vec
    if assertion:
        True_inv = inv(fH(A)@A + np.diag(D))
        assert np.allclose(True_inv, Wood_inv)
    return Wood_inv


def run_avg(a, window=25):
    return np.array([np.mean(a[i:i+window]) for i in range(len(a)-window)])

def convertFloattoPrec(f, p=4):
    return (int(f*(10**p)))/10**p