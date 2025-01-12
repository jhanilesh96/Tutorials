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

np.random.seed(87346)

_CN = myutils._CN
nmse = myutils.nmse
ctype = myutils.ctype
ftype = myutils.ftype
markov = myutils.markov1D
woodburyInv = myutils.woodburyInv
debug = 2;
cfp = myutils.convertFloattoPrec


#%% System Parameters


class Empty():
    def __init__(self):
        pass;

self = Empty()

class GenData():
    def __init__(self, K, N, VAR_N, VAR_B, A, p01, p10, logVerbose=0):
        self.K=K;
        self.N=N;
        self.VAR_N=VAR_N;
        self.VAR_B=VAR_B;
        self.logData = None;
        self.logChange = None;
        self.A = A
        self.p01 = p01
        self.p10 = p10
        self.p_avg = p01/(p10+p01)
        self.logVerbose = logVerbose
    def log(self,instruction=None,data=None):
        if self.logVerbose==0:
            return;
        if instruction is None:
            return;
        if instruction=='logData':
            if self.logData is None:
                self.logData = []
            self.logData.append(lastData);
    def sampleData(self, mode=1):
        while True:
            _s = markov(n=self.N, p01=self.p01, p10=self.p10)
            __sum = np.sum(_s)
            if __sum>0 and __sum< self.N*(self.p_avg*1.5):
                break;
        _b = np.sqrt(self.VAR_B)* _CN([self.N])
        _e = _s*_b
        _y_noiseless = self.A@_e
        _n = np.sqrt(self.VAR_N)*_CN([self.K])
        _y = _y_noiseless + _n
        lastData = [_s, _b, _e, _n, _y_noiseless, self.A, _y]
        self.log(instruction='logData', data=lastData)
        if mode==0:
            return _y, self.A
        elif mode==1:
            return _y, self.A, _e
        elif mode==2:
            return lastData


# Let us consider a Bayesian method. We will use Hidden Markov Model based approach

'''
The HMM  model assumes a Markov distribution for the support 
and a zero mean, with Gamma prior emmision distribution

i.e.
\sigma_1^2 ~ Gamma(alpha1, beta1), t_m = 1 and (~ __VAR_N + __VAR_B)
\sigma_0^2 ~ Gamma(alpha0, beta0), when t_m = 0 (~__VAR_N)

Here \sigma_0^2 corresponds to the noise variance and \sigma_1^2 is the sum of the noise and channel variance


prior distribution::::
    p(s) = MM(N, p01, p10)
    p(l_n|s_n; beta1) = s_n Gamma(alpha1, beta1) + (1-s_n) Gamma(alpha0, beta0)
    p(h_n|l_n) = CN(0,l_n^-1)
    p(y|h; betaz) = CN(Ah, z^-1);

Posterior distribution:
    q(s) = bernoulli(q_local_s)
    q(l_n) = Gamma(q_local_l_alpha, q_local_l_beta)
    q(h_n) = CN(q_local_h_mu, q_local_h_sigma)

hyperprior distribution::::
    p(beta1) = Gamma(alpha1i, beta1i)
    p(betaz) = Gamma(alphazi, betazi)
    p01 = Beta(alpha01i, beta01i)
    p10 = Beta(alpha10i, beta10i)

Posterior distribution:
    q(beta1) = Gamma(q_global_beta1_alpha, q_global_beta1_beta)
    q(betaz) = Gamma(q_global_betaz_alpha, q_global_betaz_beta)
    q01 = Beta(q_global_p01_alpha, q_global_p01_beta)
    q10 = Beta(q_global_p10_alpha, q_global_p10_beta)

We fix alpha0, alpha1, alphaz as conjugate hyperprior of alpha parameter is complicated and unnecessary.
We also fix beta0 with alpha0i/beta0i as a very small value to obtain good convergence for parameters 
of beta1 and betaz
'''
class HMM_Module_E():
    def __init__(self):
        pass;
    def infer(self,y, A, params, debugParams=None, ITER=300, tf=None):
        if debug>=2:
            dyy, dss, dbb, dee = debugParams
        # y = A h
        ''' input params from M module'''
        if params['type']=='BAYESIAN_ESTIMATE': # BAYESIAN
            P01 = params['p01']
            P10 = params['p10']
            alpha1 = params['alpha1'] # FIXED
            beta1 = params['beta1'] 
            alpha0 = params['alpha0']  # FIXED
            beta0 = params['beta0']  # FIXED
            alphaz = params['alphaz']  # FIXED
            betaz = params['betaz']
            VAR1 = beta1/alpha1
            VAR0 = beta0/alpha0
            VARN = alphaz/betaz
        ''' inferred params for local inference'''
        P00 = 1-P01
        P11 = 1-P10
        PI = P01/(P01+P10)
        ''' some useful functions and computations '''        
        K, N = A.shape
        fH = lambda _X : np.conjugate(np.transpose(_X))
        eye = lambda _M : np.eye(*[_M,_M], dtype='complex128')
        gamma = scipy.special.gamma
        digamma = scipy.special.digamma
        inv = np.linalg.inv
        logit = scipy.special.logit
        expit = scipy.special.expit
        log = np.log
        AAH = A@fH(A)
        AHA = fH(A)@A
        AbysqrtVARN = A/np.sqrt(VARN)
        LOG_NORMAL_CONST = lambda sigma_2, _N=1 : (-0.5*_N)*log(2*np.pi*sigma_2)
        IDX_OF_q_local_h_mu = 0
        ''' 1. init function, can use better initialization such as via ridge regression or via NN to successively initialize parameters '''
        def VB_init(doRidge=False):
            if doRidge:
                # From p(h_n|l_n)
                q_local_h_mu = np.linalg.inv(AHA/VARN + 0.1*eye(N))@((fH(A)@y)/VARN)
                q_local_h_sigma = np.diag(np.abs(q_local_h_mu)**2);
                q_local_s = 0.9*((np.abs(q_local_h_mu)**2>6*VARN) + (1-0.9)/0.9) 
                q_local_l_alpha = q_local_s;
                q_local_l_beta = q_local_s*(2*VAR1); # alpha/beta = q_local_s*VAR_B**-1
            else:
                q_local_h_mu = np.zeros(N, dtype=ctype)
                q_local_h_sigma = 0.001*eye(N);
                q_local_s = np.ones(N, dtype=ftype) * PI
                q_local_l_alpha = q_local_s;
                q_local_l_beta = q_local_s*VAR1; # alpha/beta = q_s*VAR_B**-1
            localVars = [q_local_h_mu, q_local_h_sigma, q_local_l_alpha, q_local_l_beta, q_local_s]
            return localVars
        ''' 2. update Functions '''
        ''' 
            2.a q(S) 
            q(S) \propto exp[ E_{q(l_n)} log[p(s)p(l_n|s_n)] ]
        '''
        def update_qS(localVars, updateOdd=True, _BGP=False):
            [q_local_h_mu, q_local_h_sigma, q_local_l_alpha, q_local_l_beta, q_local_s] = localVars
            prev_q_local_s = np.copy(q_local_s)
            # # From s_n Gamma(alpha1, beta1)
            # q_local_s = alpha1*log(beta1) - log(gamma(alpha1)) \
            #     + (alpha1-1)*(digamma(q_local_l_alpha) - log(q_local_l_beta))\
            #         - beta1 * (q_local_l_alpha/q_local_l_beta)
            # # From (1-s_n) Gamma(alpha0, beta0)
            # q_local_s -= alpha0*log(beta0) - log(gamma(alpha0)) \
            #     + (alpha0-1)*(digamma(q_local_l_alpha) - log(q_local_l_beta))\
            #         - beta0 * (q_local_l_alpha/q_local_l_beta)
            # Numerically Stable
            # From s_n Gamma(alpha1, beta1)
            q_local_s = alpha1*log(beta1) \
                + (alpha1-1)*(digamma(q_local_l_alpha) - log(q_local_l_beta))\
                    - beta1 * (q_local_l_alpha/q_local_l_beta)
            # From (1-s_n) Gamma(alpha0, beta0)
            q_local_s -= alpha0*log(beta0) \
                + (alpha0-1)*(digamma(q_local_l_alpha) - log(q_local_l_beta))\
                    - beta0 * (q_local_l_alpha/q_local_l_beta)
            # log(gamma(alpha1)) - log(gamma(alpha0)) ~~ log(alpha0^(alpha1-alpha0)) = (alpha1-alpha0)*log(alpha0)
            if np.min([np.abs(alpha1), np.abs(alpha0)])>=10:
                q_local_s -= (alpha1-alpha0)*log(alpha0) 
            else:
                q_local_s -= log(gamma(alpha1)) - log(gamma(alpha0))
            # From MM(N, p01, p10)
            if _BGP:
                q_local_s += logit(PI)
            else:
                idxOdd = np.arange(0,N,2)
                idxEven = np.arange(1,N,2)
                temp_q_local_s = np.copy(q_local_s)
                # factor from n-1
                temp_q_local_s[1:] += prev_q_local_s[:N-1]*(log(P11) - log(P10)) + (1-prev_q_local_s[:N-1])*(log(P01) - log(P00))
                # factor from n+1
                temp_q_local_s[:N-1] += prev_q_local_s[1:]*(log(P11) - log(P01)) + (1-prev_q_local_s[1:])*(log(P10) - log(P00))
                # factor for index 0
                temp_q_local_s[0] += logit(PI)
                if updateOdd:
                    q_local_s[idxOdd] = temp_q_local_s[idxOdd]
                else:
                    q_local_s[idxEven] = temp_q_local_s[idxEven]
            q_local_s = expit(q_local_s)
            # return localVars
            localVars = [q_local_h_mu, q_local_h_sigma, q_local_l_alpha, q_local_l_beta, q_local_s]
            return localVars
        ''' 
            2.b q(l) 
            q(l_n) \propto exp[ E_{q(s_n)q(h_n)} log[p(l_n|s_n)p(h_n|l_n)] ]
        '''
        def update_qL(localVars):
            [q_local_h_mu, q_local_h_sigma, q_local_l_alpha, q_local_l_beta, q_local_s] = localVars
            # (alpha-1) is the coefficient of log(x) in the log Gamma distribution
            # -beta is the coefficient of x in the log Gamma distribution
            # From s_n Gamma(alpha1, beta1) 
            q_local_l_alpha =  q_local_s*(alpha1-1)
            q_local_l_beta =  q_local_s*beta1
            # From 1-s_n Gamma(alpha1, beta1)
            q_local_l_alpha += (1-q_local_s)*(alpha0-1)
            q_local_l_beta += (1-q_local_s)*beta0
            # From CN(0,l_n^-1) 
            # if using real multivariate normal then q_local_l_F=0.5
            q_local_l_F = 1.0
            q_local_l_alpha += q_local_l_F
            q_local_l_beta += q_local_l_F*(np.abs(q_local_h_mu)**2 + np.real(np.diagonal(q_local_h_sigma)))
            # Repeat: (alpha-1) is the coefficient of log(x) in the log Gamma distribution
            q_local_l_alpha += 1
            # return localVars
            localVars = [q_local_h_mu, q_local_h_sigma, q_local_l_alpha, q_local_l_beta, q_local_s]
            return localVars
        ''' 
            2.c q(h) 
            q(h) \propto exp[ E_{q(l_n)} log[p(h_n|l_n)p(y|h)] ]
        '''
        def update_qH(localVars):
            [q_local_h_mu, q_local_h_sigma, q_local_l_alpha, q_local_l_beta, q_local_s] = localVars
            # From p(y|h) and From p(h_n|l_n)
            # q_local_h_sigma_inv = AHA/VARN + np.diag(q_local_l_alpha/q_local_l_beta);
            # q_local_h_sigma = inv(q_local_h_sigma_inv)
            # np.allclose(woodburyInv(D, A), inv(fH(A)@A + np.diag(D)))
            q_local_h_sigma = woodburyInv(q_local_l_alpha/q_local_l_beta, AbysqrtVARN)
            q_local_h_mu = q_local_h_sigma@((fH(A)@y)/VARN)
            # return localVars
            localVars = [q_local_h_mu, q_local_h_sigma, q_local_l_alpha, q_local_l_beta, q_local_s]
            return localVars
        ''' 3. Algorithm '''
        localVars = VB_init()
        q_local_h_mu_prev = localVars[IDX_OF_q_local_h_mu]
        for iters in range(ITER):
            # Update q(L)
            if debug>=2:
                dyy, dss, dbb, dee = debugParams
                [q_local_h_mu, q_local_h_sigma, q_local_l_alpha, q_local_l_beta, q_local_s] = localVars
                if iters==0:
                    q_local_h_mu = dee
                    q_local_h_sigma = np.diag(0.01 + 1e-5+np.abs(dee)**2)
                    q_local_s = np.copy(dss)
                localVars = [q_local_h_mu, q_local_h_sigma, q_local_l_alpha, q_local_l_beta, q_local_s]
                localVars = update_qL(localVars)
                [q_local_h_mu, q_local_h_sigma, q_local_l_alpha, q_local_l_beta, q_local_s] = localVars
                print('DEBUG:: q(L) :: 1s', (q_local_l_beta/q_local_l_alpha)[dss>0.1][:3])
                print('DEBUG:: q(L) :: 0s', (q_local_l_beta/q_local_l_alpha)[dss<0.1][:3])
            else:
                localVars = update_qL(localVars)
            # Update q(H)
            if debug>=2:
                localVars = update_qH(localVars)
                [q_local_h_mu, q_local_h_sigma, q_local_l_alpha, q_local_l_beta, q_local_s] = localVars
                print('DEBUG:: q(H) :: mu', np.linalg.norm(q_local_h_mu-dee)**2/np.linalg.norm(dee)**2)
            else:
                localVars = update_qH(localVars)
            # Update q(S)
            if debug>=2:
                localVars = update_qS(localVars, updateOdd=True)
                localVars = update_qS(localVars, updateOdd=False)
                [q_local_h_mu, q_local_h_sigma, q_local_l_alpha, q_local_l_beta, q_local_s] = localVars
                print('DEBUG:: q(S) :: 1s', q_local_s[dss>0.1][:3])
                print('DEBUG:: q(S) :: 0s', q_local_s[dss<0.1][:3])
            else:
                localVars = update_qS(localVars, updateOdd=True)
                localVars = update_qS(localVars, updateOdd=False)
            # termination condition
            q_local_h_mu = localVars[IDX_OF_q_local_h_mu]
            if np.linalg.norm(q_local_h_mu- q_local_h_mu_prev)<1e-6:
                break;
            q_local_h_mu_prev = localVars[IDX_OF_q_local_h_mu]
        [q_local_h_mu, q_local_h_sigma, q_local_l_alpha, q_local_l_beta, q_local_s] = localVars
        return q_local_h_mu, [localVars, iters]


class HMM_Module_M():
    def __init__(self, vbie, alphaInits, initParams=None, m_str=25, factor=1, rho_base=0.98, update='svn'):
        self.vbie = vbie;
        self.m_str=m_str
        self.update = update
        self.initParams = initParams
        self.alphaInits = alphaInits
        self.initGlobalParams()
        self.params_curr = self.getParamsDict();
        pass;
    def initGlobalParams(self):
        self.step=1
        self.q_global_p01_alpha = self.m_str*(0.1 if self.initParams is None else self.initParams['p01'])
        self.q_global_p01_beta = self.m_str*(1-(0.1 if self.initParams is None else self.initParams['p01']))
        self.q_global_p10_alpha = self.m_str*(0.9 if self.initParams is None else self.initParams['p10'])
        self.q_global_p10_beta = self.m_str*(1-(0.9 if self.initParams is None else self.initParams['p10']))
        self.alpha0 = alphaInits[0] if self.initParams is None else self.initParams['alpha0']
        self.beta0 = 1e-5*alphaInits[0] if self.initParams is None else self.initParams['beta0']
        self.alpha1 = alphaInits[1] if self.initParams is None else self.initParams['alpha1']
        temp_beta1 = 5.0*alphaInits[1] if self.initParams is None else self.initParams['beta1']
        self.alphaz = alphaInits[2] if self.initParams is None else self.initParams['alphaz']
        temp_betaz = (1/1e+1)*alphaInits[2] if self.initParams is None else self.initParams['betaz']
        # learnable beta1 with Gamma distribution
        self.q_global_beta1_beta = self.m_str*1.0
        self.q_global_beta1_alpha = self.q_global_beta1_beta * temp_beta1;
        # learnable betaz with Gamma distribution
        self.q_global_betaz_beta = self.m_str;
        self.q_global_betaz_alpha = self.q_global_betaz_beta * temp_betaz;
    def getParamsDict(self):
        if not hasattr(self, 'params_curr'):
            self.params_curr = {'type':'BAYESIAN_ESTIMATE'};
        self.params_curr['p01'] = self.q_global_p01_alpha/(self.q_global_p01_alpha + self.q_global_p01_beta)
        self.params_curr['p10'] = self.q_global_p10_alpha/(self.q_global_p10_alpha + self.q_global_p10_beta)
        self.params_curr['alpha1'] = self.alpha1
        self.params_curr['beta1'] = self.q_global_beta1_alpha/self.q_global_beta1_beta
        self.params_curr['alpha0'] = self.alpha0
        self.params_curr['beta0'] = self.beta0
        self.params_curr['alphaz'] = self.alphaz
        self.params_curr['betaz'] = self.q_global_betaz_alpha/self.q_global_betaz_beta      
        return self.params_curr
    def m_step(self, y, A, localVars, update, tf=None):
        fH = lambda _X : np.conjugate(np.transpose(_X))
        K, N = A.shape
        [q_local_h_mu, q_local_h_sigma, q_local_l_alpha, q_local_l_beta, q_local_s] = localVars
        if 's' in update:
            # q(p01) = exp E_{q(s_n)} log p(p01) sum_n log[p(s|p01)]
            self.q_global_p01_alpha = self.q_global_p01_alpha + np.sum((1-q_local_s[:-1])*q_local_s[1:])
            self.q_global_p01_beta = self.q_global_p01_beta + np.sum((1-q_local_s[:-1])*(1-q_local_s[1:]))
            # q(p10) = exp E_{q(s_n)} log p(p10) sum_n log[p(s|p10)]
            self.q_global_p10_alpha = self.q_global_p10_alpha + np.sum(q_local_s[:-1]*(1-q_local_s[1:]))
            self.q_global_p10_beta = self.q_global_p10_beta + np.sum(q_local_s[:-1]*q_local_s[1:])
        if 'v' in update:
            # q(beta1) = exp E_{q(l_n)q(s_n)} [log p(beta1) sum_n logp(l_n|s_n; beta1)]
            self.q_global_beta1_alpha = self.q_global_beta1_alpha + np.sum(self.alpha1*q_local_s)
            self.q_global_beta1_beta = self.q_global_beta1_beta \
                + np.sum(q_local_s*(q_local_l_alpha/q_local_l_beta))
        if 'n' in update:
            # q(betaz) = exp E_{q(h)} log p(betaz) p(y|h; betaz)
            # if using real multivariate normal then q_local_l_F=0.5
            q_local_l_F = 1.0 
            self.q_global_betaz_alpha = self.q_global_betaz_alpha + q_local_l_F*K
            self.q_global_betaz_beta = np.real(self.q_global_betaz_beta \
                + (1/self.alphaz)*q_local_l_F*(np.sum(np.abs(y-A@q_local_h_mu)**2) \
                    + np.trace((A@q_local_h_sigma@fH(A))) ))
        self.step = self.step+1
        self.params_curr = self.getParamsDict();
    def infer(self, y, A):
        q_local_h_mu, [localVars, iters] = self.vbie.infer(y, A, self.params_curr);
        [q_local_h_mu, q_local_h_sigma, q_local_l_alpha, q_local_l_beta, q_local_s] = localVars
        self.m_step(y, A, localVars, update=self.update);
        return q_local_h_mu, [localVars, iters, self.params_curr]
        





class GA_MMSE():
    def __init__(self):
        pass;
    def infer(self, y, A, s, VARN, VARB):
        # y = A[:,s]@x[s]
        K, N  = A.shape
        s_idx = s>0.1;
        fH = lambda _X : np.conjugate(np.transpose(_X))
        eye = lambda _M : np.eye(*[_M,_M], dtype='complex128')
        inv = np.linalg.inv
        h_hat = np.zeros(N, dtype=ctype)
        h_hat[s_idx] = inv(fH(A[:,s_idx])@A[:,s_idx] + (VARN/VARB)*eye(np.sum(s_idx)))@fH(A[:,s_idx])@y
        return h_hat






if __name__ == '__main__':
    __K=50
    __N=100
    __p = 0.1
    __p01 = 0.0277 # 1/__p = 1 + __p10/__p01 => __p10/(1/__p-1) = 
    __p10 = 0.25 # (__p01/__p)-__p01 
    __pi = __p01/(__p01+__p10)
    # this is not the true SNR as it depends on the average sparsity as well. 
    # Just a figure for computing noise variance for demonstration, (Actual SNR is lower (VAR_B*__pi)/VAR_N )
    __SNR = 20 
    __VAR_N = 10**(-__SNR/10)  
    __VAR_B = 1.0

    __A = np.linalg.svd(_CN([__N,__N]))[0][:__K,:__N]
    __A = _CN([__K,__N])


    # set params
    alphaInits = [0.9,5.0,2.0]
    alphaInits = [1.0,2.0,1.0]
    params_genie_aided = {}
    params_genie_aided['type']='BAYESIAN_ESTIMATE'
    params_genie_aided['p01'] = __p01
    params_genie_aided['p10'] = __p10
    params_genie_aided['alpha0'] =  alphaInits[0]
    params_genie_aided['beta0'] = alphaInits[0]*1e-4
    params_genie_aided['alpha1'] = alphaInits[1]
    params_genie_aided['beta1'] = alphaInits[1]*__VAR_B
    params_genie_aided['alphaz'] = alphaInits[2]*__VAR_N
    params_genie_aided['betaz']= alphaInits[2]


    # gen data


    genData = GenData(K=__K, N=__N, VAR_N=__VAR_N, VAR_B=__VAR_B, A=__A, p01=__p01, p10=__p10, logVerbose=0)
    ga_mmse = GA_MMSE();
    vbie = HMM_Module_E();
    vbiem_50 = HMM_Module_M(vbie, alphaInits=alphaInits, m_str=50, update='svn');
    vbiem_250 = HMM_Module_M(vbie, alphaInits=alphaInits, m_str=250, update='svn');
    vbiem_1000 = HMM_Module_M(vbie, alphaInits=alphaInits, m_str=1000, update='svn');
    # vbiem_ga = HMM_Module_M(vbie, initParams=params_genie_aided, alphaInits=alphaInits, update='', m_str=1);

    debug=1
    nmse_e  = []; s_e = [];
    nmse_ga = []; s_ga = [];
    nmse_em_50 = []; s_em_50 = [];
    nmse_em_250 = []; s_em_250 = [];
    nmse_em_1000 = []; s_em_1000 = [];
    # nmse_em_ga = []; s_em_ga = [];
    params=[]
    print('True:', __p01, __p10, __VAR_B, __VAR_N, sep='\t')
    params_em_250 = vbiem_250.params_curr
    print('-1:',cfp(params_em_250['p01']), cfp(params_em_250['p10']), cfp(params_em_250['beta1']/params_em_250['alpha1']), \
            params_em_250['alphaz']/cfp(params_em_250['betaz']), sep='\t');
    for _i in range(500):
        print(_i, end='\t', flush=True);
        dss, dbb, dee, _n, _y_noiseless, dAA, dyy = genData.sampleData(mode=2)
        #
        q_local_h_mu_em_50, [localVars_em_50, iters_em_50, params_em_50] = vbiem_50.infer(dyy, dAA)
        nmse_em_50.append(nmse(dee, q_local_h_mu_em_50)); 
        s_em_50.append(np.sum(np.logical_xor(dss, localVars_em_50[-1]>0.5))/__N)
        #
        q_local_h_mu_em_250, [localVars_em_250, iters_em_250, params_em_250] = vbiem_250.infer(dyy, dAA)
        nmse_em_250.append(nmse(dee, q_local_h_mu_em_250)); 
        s_em_250.append(np.sum(np.logical_xor(dss, localVars_em_250[-1]>0.5))/__N)
        #
        q_local_h_mu_em_1000, [localVars_em_1000, iters_em_1000, params_em_1000] = vbiem_1000.infer(dyy, dAA)
        nmse_em_1000.append(nmse(dee, q_local_h_mu_em_1000)); 
        s_em_1000.append(np.sum(np.logical_xor(dss, localVars_em_1000[-1]>0.5))/__N)
        #
        q_local_h_mu, [localVars_e, iters] = vbie.infer(dyy, dAA, params=params_genie_aided, debugParams=[dyy, dss, dbb, dee], ITER=300)
        nmse_e.append(nmse(dee, q_local_h_mu)); 
        s_e.append(np.sum(np.logical_xor(dss, localVars_e[-1]>0.5))/__N)
        #
        q_local_h_mu_ga_mmse = ga_mmse.infer(dyy, dAA, dss, __VAR_N, __VAR_B)
        nmse_ga.append(nmse(dee, q_local_h_mu_ga_mmse)); 
        s_ga.append(0)
        #
        # q_local_h_mu_em_ga, [localVars_em_ga, iters_em_ga, params_em_ga] = vbiem_ga.infer(dyy, dAA)
        # nmse_em_ga.append(nmse(dee, q_local_h_mu_em_ga)); s_em_ga.append(np.sum(np.logical_xor(dss, localVars_em_ga[-1]>0.5)))
        # prints
        print(cfp(params_em_250['p01']), cfp(params_em_250['p10']), cfp(params_em_250['beta1']/params_em_250['alpha1']), \
            params_em_250['alphaz']/cfp(params_em_250['betaz']), '','S:', 'A'+str(np.mean(dss)), s_em_250[-1], s_e[-1], \
                'N:', cfp(nmse_em_250[-1]), cfp(nmse_e[-1]), cfp(nmse_ga[-1]), sep='\t', end='', flush=True);
        print()
        if (_i+1)%50==0:
            print('True:', __p01, __p10, __VAR_B, __VAR_N, sep='\t')
            plt.close('all')
            plt.semilogy(myutils.run_avg(nmse_em_50), label='EM_50');
            plt.semilogy(myutils.run_avg(nmse_em_250), label='EM_250');
            plt.semilogy(myutils.run_avg(nmse_em_1000), label='EM_1000');
            plt.semilogy(myutils.run_avg(nmse_e), label='E');
            plt.semilogy(myutils.run_avg(nmse_ga), label='GA');
            plt.legend()
            plt.savefig('./convergence_nmse')



    # print('VBIE   :', np.mean(nmse_e), np.mean(s_e))
    # print('VBIEM  :', np.mean(nmse_em), np.mean(s_em))
    # # print('VBIEM_GA:', np.mean(nmse_em_ga), np.mean(s_em_ga))
    # print('GA_MMSE:', np.mean(nmse_ga), np.mean(s_ga))


    # localVars = localVars_em
    # q_local_s = localVars_em[-1]
    # self=vbiem
    # y, A = dyy, dAA
    # fH = lambda _X : np.conjugate(np.transpose(_X))
    # K, N = A.shape
    # [q_local_h_mu, q_local_h_sigma, q_local_l_alpha, q_local_l_beta, q_local_s] = localVars_e
    # q_local_l_F = 1.0

    # # plt.plot(myutils.run_avg(nmse_em, 25), label='EM');
    # # plt.plot(myutils.run_avg(nmse_e, 25), label='E');
    # # plt.plot(myutils.run_avg(nmse_ga, 25), label='GA');
    # # plt.legend()
    # # plt.show()

    # # if we need ridge regression
    # # y, A = _y, __A
    # # VARN = __VAR_N
    # # fH = lambda _X : np.conjugate(np.transpose(_X))
    # # AAH = A@fH(A)
    # # AHA = fH(A)@A
    # # eye = lambda _M : np.eye(*[_M,_M], dtype='complex128')
    # # q_local_h_mu = np.linalg.inv(AHA/VARN + 0.1*eye(__N))@((fH(A)@y)/VARN)
    # # print(nmse(_e, q_local_h_mu))
    # # (np.sum((np.abs(q_local_h_mu)**2> 9*VARN)[dss>0.1]), np.sum((np.abs(q_local_h_mu)**2> 9*VARN)[dss<0.1]))







