import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# M_x = 8 
# M_y = 8
# N = 16
# DFTfactor = 1
# speed_u_min = 40
# speed_u_max = 40
# h_min = 20 
# h_max = 200
# frames = 20
# f_c = 5.8e9
# max_delta_tau = 1.6e-6
# markov_lam_01 = 0.8;
# markov_lam_10 = 0.08;
# P_base = 5
# P_inc = 5

# T_c = 0.423*((f_c/3e8)*speed_u_max)**-1
# _phased = False
# _uniqueActivity = True



def splitIntoMeanAndVar(_, _splits=2):
    if _splits==2:
        mean, logvar = tf.split(_, num_or_size_splits=_splits, axis=-1)
        return mean, logvar
    else:
        return tf.split(_, num_or_size_splits=_splits, axis=-1);

def reparameterize(mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean



def vectorizeChannelSample(__channel_s, _dim_x, _complex=False):
    if not _complex:
        # expected shape [None, N, M_x_ft, M_y_ft]
        return tf.reshape(__channel_s, [-1, _dim_x])
    else:
        return tf.reshape(__channel_s, [-1, 2*_dim_x])

'''
To increase this factor:
we want to decrease tf.divide(tf.square(_x-_mean), tf.exp(_logvar))
and we want to decrease _logvar
i.e 
first factor wants to
    decrease tf.square(_x-_mean)
    increase tf.exp(_logvar)
second factor wants to 
    decrease _logvar


'''
def logProbNormalUncorrelated(_x, _mean, _logvar):
    return -0.5*_logvar-0.5*np.log(2*np.pi)-0.5*tf.divide(tf.square(_x-_mean), tf.exp(_logvar))


def logProbNormalUncorrelatedComplex(_x, _mean, _logvar):
    return -0.5*_logvar-0.5*np.log(2*np.pi)-0.5*tf.divide(tf.square(tf.abs(_x-_mean)), tf.exp(_logvar))

#KL(q||P)
# mean_q, logvar_q, mean_p, logvar_p, dimz = mean_z_t_inf, logvar_z_t_inf, mean_z_t_prior, logvar_z_t_prior, dim_z
# mean_q, logvar_q, mean_p, logvar_p, dimz = mean_z_t_inf, logvar_z_t_inf, mean_z_t_inf, logvar_z_t_inf, dim_z
def kldivNormalQP(mean_q, logvar_q, mean_p, logvar_p, dimz):
    return 0.5*( tf.math.reduce_sum(logvar_p-logvar_q, axis=-1)\
            - dimz\
                + tf.reduce_sum(tf.exp(logvar_q-logvar_p),axis=-1) \
                    + tf.reduce_sum(tf.multiply(tf.multiply(tf.exp(-logvar_p), (mean_p-mean_q)), (mean_p-mean_q)), axis=-1))


# '''
# Phi_prior.trainable_weights
# correlated_channel_sample = _channel_samples;
# splitIntoMeanAndVar = commons.splitIntoMeanAndVar
# logProbNormalUncorrelated = commons.logProbNormalUncorrelated
# vectorizeChannelSample = commons.vectorizeChannelSample
# reparameterize = commons.reparameterize
# onlyMSE=False
# '''
# @tf.function
# def ForwardLosses_v2(correlated_channel_sample, mlps, params, \
#     scale_kl=1.0, scale_ll=1.0, onlyMSE=False, dropout_rate_gist=0.0, _complex=False, _complexify_version=1):
#     # https://github.com/crazysal/VariationalRNN/blob/master/VariationalRecurrentNeuralNetwork-master/model.py
#     [Phi_prior, Phi_z, Phi_dec, Phi_h, f_theta, Phi_enc] = mlps
#     [dim_x, dim_z, dim_g, Dropout] = params
#     batches, _frames = correlated_channel_sample.shape[:2]
#     dim_h = Phi_h.outputs[0].shape[-1]
#     obj_list = [] 
#     kl_list = []
#     loglike_list = []
#     g_t_1 = g_0 = tf.zeros(shape=[batches,dim_g]);
#     t=0
#     for t in range(_frames):
#         ''' Step 2.1 q(z_t | g_t_1, Phi_h(h_t)) = Phi_enc(tf.concat([ho_t,g_t_1],axis=-1)): Infer Latent Variable using Sample, h_t, from dataset and Gist, g_t_1'''
#         _shape = [-1]; _shape.extend(Phi_h.input.shape[1:])
#         _true_h_t = tf.reshape(correlated_channel_sample[:,t],_shape)
#         ho_t = Phi_h(_true_h_t)
#         mean_z_t_inf, logvar_z_t_inf = Phi_enc(tf.concat([ho_t,g_t_1],axis=-1))
#         z_t_inf = reparameterize(mean_z_t_inf, logvar_z_t_inf)
#         ''' Step 1.1 p(z_t | g_t_1) = Phi_prior(g_t_1): Prior of Latent variable '''
#         mean_z_t_prior, logvar_z_t_prior = Phi_prior(g_t_1)
#         ''' Step 1.2 p(h_t | Phi_z(z_t), g_t_1) = Phi_dec(tf.concat([zo_t,g_t_1_dropped], axis=-1)): Generate Sample using Latent Variable and dropped out Gist '''
#         g_t_1_dropped = Dropout(rate=dropout_rate_gist)(g_t_1)
#         zo_t = Phi_z(z_t_inf)
#         mean_h_t, logvar_h_t = Phi_dec(tf.concat([zo_t,g_t_1_dropped], axis=-1))
#         ''' Step 1.3 g_t = f_theta(g_t_1, Phi_h(h_t), Phi_z(z_t)): Generate Gist using True Sample and Reparameterised'''
#         g_t = f_theta(tf.concat([ho_t,zo_t,g_t_1], axis=-1))
#         ''' Step 2.1 Compute KL(q(z_t | g_t_1, Phi_h(h_t)) || p(z_t | g_t_1))'''
#         kl = 0.5*( tf.math.reduce_sum(logvar_z_t_prior-logvar_z_t_inf, axis=-1)\
#             - dim_z\
#                 + tf.reduce_sum(tf.exp(logvar_z_t_inf-logvar_z_t_prior),axis=-1) \
#                     + tf.reduce_sum(tf.multiply(tf.multiply(tf.exp(-logvar_z_t_prior), (mean_z_t_prior-mean_z_t_inf)), (mean_z_t_prior-mean_z_t_inf)), axis=-1))
#         ''' Step 2.2 Compute log p(h_t | Phi_z(z_t), g_t_1)'''
#         _arg1 = vectorizeChannelSample(correlated_channel_sample[:,t],dim_h,_complex=False)
#         _arg2 = vectorizeChannelSample(mean_h_t,dim_h,_complex=False)
#         _arg3 = vectorizeChannelSample(logvar_h_t,dim_h,_complex=False)
#         loglike = logProbNormalUncorrelated(_arg1,_arg2,_arg3)
#         loglike = tf.reduce_sum(loglike,axis=-1)
#         ''' new addition '''
#         mse_obj = tf.reduce_sum(tf.square(tf.abs(_arg1-_arg2)), axis=-1)
#         # assert np.allclose(loglike, tf.reduce_sum(loglike2,axis=-1))
#         '''EO new addition'''
#         if onlyMSE:
#             obj_t = -50*mse_obj + (-scale_kl*kl + scale_ll*loglike)
#         else:
#             obj_t = -scale_kl*kl + scale_ll*loglike
#         kl_list.append(kl)
#         loglike_list.append(loglike)
#         obj_list.append(obj_t)
#         g_t_1 = g_t
#     # maximize objective
#     objective = tf.reduce_mean(obj_list);
#     return objective, [obj_list, loglike_list, kl_list, mse_obj];



# '''
# Phi_prior.trainable_weights
# correlated_channel_sample = _channel_samples;
# splitIntoMeanAndVar = commons.splitIntoMeanAndVar
# logProbNormalUncorrelated = commons.logProbNormalUncorrelated
# vectorizeChannelSample = commons.vectorizeChannelSample
# reparameterize = commons.reparameterize
# onlyMSE=False
# scale_kl=scale_ll=0
# '''
# @tf.function
# def ForwardLosses_pretrain_v2(correlated_channel_sample, mlps, params, \
#     scale_kl=1.0, scale_ll=1.0, onlyMSE=False, dropout_rate_gist=0.0, _complex=False, _complexify_version=1):
#     # https://github.com/crazysal/VariationalRNN/blob/master/VariationalRecurrentNeuralNetwork-master/model.py
#     [Phi_prior, Phi_z, Phi_dec, Phi_h, f_theta, Phi_enc] = mlps
#     [dim_x, dim_z, dim_g, Dropout] = params
#     batches, _frames = correlated_channel_sample.shape[:2]
#     dim_h = Phi_h.outputs[0].shape[-1]
#     obj_list = [] 
#     mean_h_t_list = []
#     z_t_inf_list = []
#     g_t_list = []
#     g_t_1 = g_0 = tf.zeros(shape=[batches,dim_g]);
#     t=0
#     for t in range(_frames):
#         ''' Step 2.1 q(z_t | g_t_1, Phi_h(h_t)) = Phi_enc(tf.concat([ho_t,g_t_1],axis=-1)): Infer Latent Variable using Sample, h_t, from dataset and Gist, g_t_1'''
#         _shape = [-1]; _shape.extend(Phi_h.input.shape[1:])
#         _true_h_t = tf.reshape(correlated_channel_sample[:,t],_shape)
#         ho_t = Phi_h(_true_h_t)
#         mean_z_t_inf, _ = Phi_enc(tf.concat([ho_t,g_t_1],axis=-1))
#         z_t_inf = mean_z_t_inf
#         ''' Step 1.2 p(h_t | Phi_z(z_t), g_t_1) = Phi_dec(tf.concat([zo_t,g_t_1_dropped], axis=-1)): Generate Sample using Latent Variable and dropped out Gist '''
#         zo_t = z_t_inf#Phi_z(z_t_inf)
#         mean_h_t, _ = Phi_dec(tf.concat([zo_t,g_t_1], axis=-1))
#         ''' Step 1.3 g_t = f_theta(g_t_1, Phi_h(h_t), Phi_z(z_t)): Generate Gist using True Sample and Reparameterised'''
#         g_t = f_theta(tf.concat([ho_t,zo_t,g_t_1], axis=-1))
#         ''' Step 2.2 Compute log p(h_t | Phi_z(z_t), g_t_1)'''
#         _arg1 = vectorizeChannelSample(correlated_channel_sample[:,t],dim_h,_complex=False)
#         _arg2 = vectorizeChannelSample(mean_h_t,dim_h,_complex=False)
#         ''' new addition '''
#         mse_obj = -tf.reduce_sum(tf.square(tf.abs(_arg1-_arg2)), axis=-1)
#         # assert np.allclose(loglike, tf.reduce_sum(loglike2,axis=-1))
#         g_t_1 = g_t
#         obj_list.append(mse_obj)
#         mean_h_t_list.append(mean_h_t)
#         z_t_inf_list.append(z_t_inf)
#         g_t_list.append(g_t)
#     # maximize objective
#     objective = tf.reduce_mean(obj_list);
#     return objective, [correlated_channel_sample, mean_h_t_list, z_t_inf_list, g_t_list];














