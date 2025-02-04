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

config=2;


USE_REDUCE_VARIANCE = True;
dim_h = genData.Kx*genData.Ky
dim_z = int(0.15*dim_h); 
dim_zo = dim_z;
dim_g = dim_h//3; dim_ho = dim_h;
layers_n1 = [4*dim_z, 2*dim_z] # 1.1.1 Phi_prior ::: Dense (stochastic) ::: z_t | g_t_1 => dim_g -> 2(dim_z)
filters_n2 = [[64,32,32,16]]*2 # all blocks must end at same num of filters 1.1.2 Phi_dec ::: ConvT (stochastic) ::: h_t | Phi_z(z_t), g_t_1 => dim_zo + dim_g -> 2(dim_h)
layers_nz = [dim_z] # 1.1.2.A Phi_z ::: Dense (function) ::: z_t -> zo_t => dim_z -> dim_zo
filters_nh = [20,15,2] # 1.1.3.A Phi_h ::: Conv (function) ::: h_t -> ho_t => dim_h -> dim_ho
layers_n3 = [2*dim_g, dim_g] # 1.1.4 f_theta ::: Dense (function) ::: g_t | g_t_1, Phi_h(h_t), Phi_z(z_t) => dim_ho + dim_zo + dim_g -> dim_g
layers_n4 = [4*dim_z, 2*dim_z, 2*dim_z] # 1.2.1 Phi_enc ::: Dense (stochastic) ::: z_t | g_t_1, Phi_h(h_t) => dim_g + dim_ho -> dim_z

if config==2:
    USE_REDUCE_VARIANCE = True;


VARIANCE_MULT = 5.0
import tensorflow as tf
import tensorflow_probability as tfp
tfk = tf.keras
'''' 1. Networks '''
'''' 1.1  Generative Networks '''
'''' 1.1.1 Phi_prior ::: Dense (stochastic) ::: z_t | g_t_1 => dim_g -> 2(dim_z) '''
i = tfk.layers.Input(shape=dim_g); _ = i
for _num in layers_n1:
    _ = tfk.layers.Dense(int(_num), activation='relu')(_)
_ = tfk.layers.BatchNormalization()(_)
_m = tfk.layers.Dense(dim_z)(_)
_v = tfk.layers.Dense(dim_z)(_)
# SHORTCUT:: not required but this strategy limits search variance improving, faster training 
_v = -8.0*tf.nn.sigmoid(_v/1000)+1.0 if USE_REDUCE_VARIANCE else _v
Phi_prior_mean = tfk.Model(inputs=i, outputs=_m, name='Phi_prior_m')
Phi_prior_var = tfk.Model(inputs=i, outputs=_v, name='Phi_prior_v')
Phi_prior = tfk.Model(inputs=i, outputs=[_m,_v], name='Phi_prior')
Phi_prior.summary()

'''' 1.1.2.A Phi_z ::: Dense (function) ::: z_t -> zo_t => dim_z -> dim_zo '''
i = tfk.layers.Input(shape=dim_z); _ = i
for _num in layers_nz:
    _ = tfk.layers.Dense(int(_num), activation='relu')(_)
_ = tfk.layers.BatchNormalization()(_)
_ = tfk.layers.Dense(dim_zo)(_)
Phi_z = tfk.Model(inputs=i, outputs=_, name='Phi_z')
Phi_z.summary()

'''' 1.1.2 Phi_dec ::: ConvT (stochastic) ::: h_t | Phi_z(z_t), g_t_1 => dim_zo + dim_g -> 2(dim_h) '''
i = tf.keras.layers.Input(shape=dim_zo+dim_g);
_ = tf.keras.layers.Dense(units=genData.Kx*genData.Ky*4, activation=None)(i)
_ = tfk.layers.BatchNormalization()(_)
_0 = tf.nn.leaky_relu(_)
_ = _0
# _ = tf.keras.layers.Dense(units=genData.Kx*genData.Ky*4, activation=None)(_)
# _ = tfk.layers.BatchNormalization()(_)
# _1 = tf.nn.leaky_relu(_)
# _ = _1+_0
# _ = tf.keras.layers.Dense(units=genData.Kx*genData.Ky*4, activation=None)(_)
# _ = tfk.layers.BatchNormalization()(_)
# _2 = tf.nn.leaky_relu(_)
# _ = _2+_1
# _ = tf.keras.layers.Dense(units=genData.Kx*genData.Ky*4, activation=None)(_)
# _ = tfk.layers.BatchNormalization()(_)
# _3 = tf.nn.leaky_relu(_)
# _ = _3+_2
_ = tf.keras.layers.Reshape(target_shape=(genData.Kx, genData.Ky, 4))(_)
_init = tf.keras.layers.Conv2DTranspose(filters=filters_n2[0][-1], kernel_size=9, strides=1, padding='same')(_)
for _block in range(len(filters_n2)):
    filters_n2_block = filters_n2[_block]
    _ = _init
    for _filters in filters_n2_block:
        _ = tf.keras.layers.Conv2DTranspose(filters=_filters, kernel_size=9, strides=1, padding='same')(_)
        _ = tfk.layers.BatchNormalization()(_)
        _ = tf.nn.leaky_relu(_)
    _init = _init + _
    _init = tfk.layers.BatchNormalization()(_init)

# _ = _init
_ = tf.keras.layers.Conv2DTranspose(filters=10, kernel_size=9, strides=1, padding='same', activation=None)(_)
_m = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=9, strides=1, padding='same', activation=None)(_)
_v = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=9, strides=1, padding='same', activation=None)(_)
# SHORTCUT:: not required but this strategy limits search variance improving, faster training 
_m = 10.0*tf.nn.sigmoid(_m)
_v = -8.0*tf.nn.sigmoid(_v/1000)+1.0 if USE_REDUCE_VARIANCE else _v
Phi_dec = tfk.Model(inputs=i, outputs=[_m, _v], name='Phi_dec')
Phi_dec_mean = tfk.Model(inputs=i, outputs=_m, name='Phi_decm')
Phi_dec_var = tfk.Model(inputs=i, outputs=_v, name='Phi_decv')
Phi_dec.summary()

'''' 1.1.3.A Phi_h ::: Conv (function) ::: h_t -> ho_t => dim_h -> dim_ho '''
i = tfk.layers.Input(shape=(genData.Kx,genData.Ky)); _ = i
_ = tfk.layers.Reshape(target_shape=(genData.Kx,genData.Ky,1))(_);
for _filters in filters_nh:
    _ = tfk.layers.Conv2D(filters=_filters, padding='same', kernel_size=5, strides=(1, 1), activation='relu')(_)
_ = tfk.layers.Flatten()(_)
_ = tfk.layers.BatchNormalization()(_)
_ = tfk.layers.Dense(dim_ho)(_)
Phi_h = tfk.Model(inputs=i, outputs=_, name='Phi_h')
Phi_h.summary()

'''' 1.1.4 f_theta ::: Dense (function) ::: g_t | g_t_1, Phi_h(h_t), Phi_z(z_t) => dim_ho + dim_zo + dim_g -> dim_g '''
i = tfk.layers.Input(shape=dim_ho+dim_zo+dim_g); _ = i
for _num in layers_n3:
    _ = tfk.layers.Dense(int(_num), activation='relu')(_)
_ = tfk.layers.BatchNormalization()(_)
_ = tfk.layers.Dense(dim_g)(_)
f_theta = tfk.Model(inputs=i, outputs=_, name='f_theta')
f_theta.summary()


''''1.2 Inference Networks '''
'''' 1.2.1 Phi_enc ::: Dense (stochastic) ::: z_t | g_t_1, Phi_h(h_t) => dim_g + dim_ho -> dim_z '''
i = tfk.layers.Input(shape=dim_ho+dim_g); _ = i
for _num in layers_n4:
    _ = tfk.layers.Dense(int(_num), activation='relu')(_)
_ = tfk.layers.BatchNormalization()(_)
_m = tfk.layers.Dense(dim_z)(_)
_v = tfk.layers.Dense(dim_z)(_)
# SHORTCUT:: not required but this strategy limits search variance improving, faster training 
_v = -8.0*tf.nn.sigmoid(_v/1000)+1.0 if USE_REDUCE_VARIANCE else _v
Phi_enc = tfk.Model(inputs=i, outputs=[_m, _v], name='Phi_enc')
Phi_enc_mean = tfk.Model(inputs=i, outputs=_m, name='Phi_encm')
Phi_enc_var = tfk.Model(inputs=i, outputs=_v, name='Phi_encv')
Phi_enc.summary()



sim_epochs_pre = 250;
sim_epochs_kl = 0;
sim_epochs_ll_kl = 200;
sim_epochs_finetuning = 30;
sim_steps_per_epoch=2000;
sim_batch_size = 20;
sim_lr_pretrain = 1e-5
sim_lr = 1e-4
sim_reparam_in_pretrain = True

if config==2:
    sim_epochs_pre = 80;
    sim_epochs_ll_kl = 200;
    sim_epochs_finetuning = 100;
    sim_lr_pretrain = 1e-4
    sim_lr = 1e-4
    sim_batch_size = 5;


'''' 2. Combined Network '''
trainable_variables = []
mlps = [Phi_prior, Phi_z, Phi_dec, Phi_h, f_theta, Phi_enc]
for _mlp in mlps:
    for _trainable_variables in _mlp.trainable_variables:
        trainable_variables.append(_trainable_variables);

train_debug=1
# train_opt = tfk.optimizers.SGD(learning_rate=sim_lr, momentum=0.01, nesterov=True)
# train_opt0 = tfk.optimizers.Adam(learning_rate=sim_lr/100)
train_opt = tfk.optimizers.Adam(learning_rate=sim_lr)
train_opt_finetune = tfk.optimizers.SGD(learning_rate=sim_lr/100)
train_checkpoint = tf.train.Checkpoint(step=tf.Variable(0), optimizer=train_opt, net=mlps)
train_folder = './results_big/training_checkpoints/train_'+str(config)
train_manager = tf.train.CheckpointManager(train_checkpoint, train_folder, max_to_keep=5)

train_pre_opt = tfk.optimizers.Adam(learning_rate=sim_lr_pretrain)
train_pre_checkpoint = tf.train.Checkpoint(step=tf.Variable(0), optimizer=train_pre_opt, net=mlps)
train_pre_folder = './results_big/training_checkpoints/train_pre_'+str(config)
train_pre_manager = tf.train.CheckpointManager(train_pre_checkpoint, train_pre_folder, max_to_keep=5)




import c_commons as commons
def _GenerateSamples(mlps, batches=10, _frames=genData.T+5):
    [Phi_prior, Phi_z, Phi_dec, Phi_h, f_theta, Phi_enc] = mlps
    h_list = [] 
    g_t_1 = g_0 = 1e-2*tf.ones(shape=[batches,dim_g]);
    for t in range(_frames):
        # Step 1.
        mean_z_t_prior, logvar_z_t_prior = Phi_prior(g_t_1)
        # mean_z_t_prior, logvar_z_t_prior = splitIntoMeanAndVar(z_t_prior_unsplit)
        z_t = commons.reparameterize(mean_z_t_prior, logvar_z_t_prior)
        # Step 2.
        zo_t = z_t# zo_t = Phi_z(z_t)
        mean_h_t, logvar_h_t = Phi_dec(tf.concat([zo_t,g_t_1], axis=-1))
        ho_t = Phi_h(mean_h_t)
        g_t = f_theta(tf.concat([ho_t,zo_t,g_t_1], axis=-1))
        g_t_1 = g_t
        h_list.append(mean_h_t)
    return h_list


GenerateSamples = lambda: _GenerateSamples(mlps=mlps)
try:
    os.makedirs('./results_big/Logs/train_'+str(config)+'/')
except:
    pass;

def savefigs(_epoch_=None):
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings("ignore")
    samples = GenerateSamples();
    plt.figure(figsize=[19.2,12]);
    plt.subplot(len(samples), samples[0].shape[0], samples[0].shape[0]*len(samples))
    for _frame in range(len(samples)):
        for _sample in range(samples[0].shape[0]):
            _ = plt.subplot(len(samples), samples[0].shape[0], _frame*samples[0].shape[0]+_sample+1)
            _ = plt.imshow(samples[_frame][_sample].numpy().reshape([genData.Kx,genData.Ky]))
            _ = plt.ylabel(str(_sample)+'_'+str(_frame))
    if _epoch_ is not None:
        plt.savefig('./results_big/Logs/train_'+str(config)+'/epoch_'+'{:05d}'.format(_epoch_))
    plt.savefig('./results_big/Logs/train_'+str(config)+'/epoch_current')
    plt.figure(figsize=[4*6.4, 4*4.8]);
    plt.close('all')



# import c_commons as commons
# params = [dim_h, dim_z, dim_g, tfk.layers.Dropout]
# gen_params = [dim_h, dim_z, dim_g]
# ForwardLosses = lambda sample, scale_kl, scale_ll, dropout_rate_gist, onlyMSE : commons.ForwardLosses_v2(sample, mlps, params=params, scale_kl=scale_kl, scale_ll=scale_ll, dropout_rate_gist=dropout_rate_gist, onlyMSE=onlyMSE);
# ForwardLosses_pretrain = lambda sample, scale_kl, scale_ll, dropout_rate_gist, onlyMSE : commons.ForwardLosses_pretrain_v2(sample, mlps, params=params, scale_kl=scale_kl, scale_ll=scale_ll, dropout_rate_gist=dropout_rate_gist, onlyMSE=onlyMSE);
# GenerateSamples = lambda : commons.GenerateSamples(mlps=mlps, params=gen_params, batches=10, _frames=genData.T+5)
# powpow = lambda x : np.power(10,x)



# savefigs()













''' Data Generator '''
def dataGenerator(_batch_size, num=1):
    while True:
        H = genData.sample(_batch_size)
        H = H.reshape(H.shape[0],H.shape[1], H.shape[2], H.shape[3])
        yield H, [H]*num




# ''' Pre-training, without any stochastic operation (same as autoencoder) '''

''' Load Last Checkpoint '''
try:
    status = train_pre_checkpoint.restore(train_pre_manager.latest_checkpoint)
    train_pre_baseEpoch = int(train_pre_manager.latest_checkpoint.split('-')[-1]) if train_pre_manager.latest_checkpoint is not None else 0
except:
    train_pre_baseEpoch = 0;

print('\n\n\n\n\n\n')
print('Pre-Loaded::', train_pre_baseEpoch)
print('\n\n\n\n\n\n')

h_in = tfk.layers.Input([genData.T,genData.Kx,genData.Ky])
g_t_1 = g_0 = 1e-2*tf.ones(shape=[sim_batch_size,dim_g]);
t=0
sample_h_t_list = []
zo_t_list = []
ll_list = []
kl_list = []
z_mean_diff_t_list = []
z_var_diff_t_list = []
z_var_sum_t_list = []
for t in range(genData.T):
    ''' Step 2.1 q(z_t | g_t_1, Phi_h(h_t)) = Phi_enc(tf.concat([ho_t,g_t_1],axis=-1)): Infer Latent Variable using Sample, h_t, from dataset and Gist, g_t_1'''
    ho_t = Phi_h(h_in[:,t])
    mean_z_t_inf = Phi_enc_mean(tf.concat([ho_t,g_t_1],axis=-1))
    logvar_z_t_inf = Phi_enc_var(tf.concat([ho_t,g_t_1],axis=-1))
    if sim_reparam_in_pretrain:
        z_t_inf = commons.reparameterize(mean_z_t_inf, logvar_z_t_inf)
    else:
        z_t_inf = mean_z_t_inf
    ''' Step 1.2 p(h_t | Phi_z(z_t), g_t_1) = Phi_dec(tf.concat([zo_t,g_t_1_dropped], axis=-1)): Generate Sample using Latent Variable and dropped out Gist '''
    mean_z_t_prior = Phi_prior_mean(g_t_1)
    logvar_z_t_prior = Phi_prior_var(g_t_1)
    zo_t = z_t_inf#Phi_z(z_t_inf)
    mean_h_t = Phi_dec_mean(tf.concat([zo_t,g_t_1], axis=-1))
    logvar_h_t = Phi_dec_var(tf.concat([zo_t,g_t_1], axis=-1))
    if sim_reparam_in_pretrain:
        sample_h_t = commons.reparameterize(mean_h_t, logvar_h_t)
    else:
        sample_h_t = mean_h_t
    ''' Step 1.3 g_t = f_theta(g_t_1, Phi_h(h_t), Phi_z(z_t)): Generate Gist using True Sample and Reparameterised'''
    g_t = f_theta(tf.concat([ho_t,zo_t,g_t_1], axis=-1))
    ''' Step 2.2 Compute log p(h_t | Phi_z(z_t), g_t_1)'''
    _arg1 = tf.reshape(h_in[:,t], [sim_batch_size,genData.Kx*genData.Ky])
    _arg2 = tf.reshape(mean_h_t, [sim_batch_size,genData.Kx*genData.Ky])
    _arg3 = tf.reshape(logvar_h_t, [sim_batch_size,genData.Kx*genData.Ky])
    loglike = commons.logProbNormalUncorrelated(_arg1,_arg2,_arg3)
    kl = commons.kldivNormalQP(mean_z_t_inf, logvar_z_t_inf, mean_z_t_prior, logvar_z_t_prior, dim_z)
    ###########################
    sample_h_t_list.append(sample_h_t)
    zo_t_list.append(zo_t)
    ll_list.append(tf.reduce_mean(loglike, axis=-1))
    kl_list.append(kl)
    # SHORTCUT:: not required but this strategy limits search variance improving, faster training 
    z_mean_diff_t_list.append(tf.reduce_mean(tf.square(mean_z_t_inf-mean_z_t_prior), axis=-1))
    z_var_diff_t_list.append(tf.reduce_mean(tf.exp(logvar_z_t_inf-logvar_z_t_prior), axis=-1))
    z_var_sum_t_list.append(tf.reduce_mean(tf.exp(logvar_z_t_inf)+tf.exp(logvar_z_t_prior), axis=-1))
    g_t_1 = g_t

# ELBO = LL-KL
# negELBO = KL-LL
ll_loss = tf.math.negative(tf.reduce_sum(tf.stack(ll_list, axis=1), axis=1), name='ll')
kl_loss = tf.reduce_sum(tf.stack(kl_list, axis=1), axis=1, name='kl')
h_out = tf.stack(sample_h_t_list, axis=1)[:,:,:,:,0]
z_out = tf.stack(zo_t_list, axis=1)
z_mean_diff_out = tf.reduce_mean(tf.stack(z_mean_diff_t_list, axis=1), axis=-1) # BxT -> B
z_var_diff_out = tf.reduce_mean(tf.stack(z_var_diff_t_list, axis=1), axis=-1) # BxT -> B
z_var_sum_out = tf.reduce_mean(tf.stack(z_var_sum_t_list, axis=1), axis=-1) # BxT -> B
z_diff = z_mean_diff_out + z_var_diff_out #+ z_var_sum_out


_metric_mse = tfk.losses.MeanSquaredError()
class MyModel(tfk.Model):
    def compute_metrics(self, x, y, y_pred, sample_weight):
        # metric_results = super(MyModel, self).compute_metrics(x, y, y_pred, sample_weight)
        # self.custom_metric.update_state(x, y, y_pred, sample_weight)
        metric_results = {}
        metric_results['LL'] = _self_loss(None, y_pred[0])
        metric_results['MSE'] = _metric_mse(y[1], y_pred[1])
        metric_results['DIFF'] = _self_loss(None, y_pred[2])
        metric_results['KL'] = _self_loss(None, y_pred[3])
        return metric_results


pre_train_model_debug = tfk.Model(inputs=h_in, outputs=[h_out, z_out, z_mean_diff_out])
# pre_train_model = tfk.Model(inputs=h_in, outputs=[h_out, z_mean_diff_out])
# pre_train_model = tfk.Model(inputs=h_in, outputs=[h_out, z_mean_diff_out+z_var_diff_out, h_out])
pre_train_model = MyModel(inputs=h_in, outputs=[ll_loss, h_out, z_diff, kl_loss])
_self_loss=lambda y_true, y_pred: y_pred
_nmse_loss=lambda y_true, y_pred: tf.reduce_sum(((y_pred-y_true)**2), axis=(-1,-2,-3))/tf.reduce_sum(((y_true)**2), axis=(-1,-2,-3))
_losses  = [_self_loss, 'mse', _self_loss, _self_loss]
_weights = [1.0, 500.0, 1.0, 0.0]
pre_train_model.compile(loss=_losses,loss_weights=_weights, optimizer=train_pre_opt, metrics=_losses)

if __name__=='__main__':
    print('Pretraining v2...')
    cb_pre = tf.keras.callbacks.Callback()
    def onEpochEnd(epoch, logs=None):
        train_pre_manager.save();
    cb_pre.on_epoch_end = onEpochEnd
    _gen = dataGenerator(_batch_size=sim_batch_size, num=len(pre_train_model.outputs))
    h_in = next(_gen)[0]
    # pre_train_model.fit(_gen, epochs=1, steps_per_epoch=100)
    # exit()
    pre_train_model.fit(_gen, epochs=sim_epochs_pre-train_pre_baseEpoch, steps_per_epoch=sim_steps_per_epoch, callbacks=[cb_pre])

pre_train_model.trainable=False
savefigs()



















# exit()


# print(Phi_dec_mean.trainable_weights[0][0][0])
try:
    status = train_checkpoint.restore(train_manager.latest_checkpoint)
    train_baseEpoch = int(train_manager.latest_checkpoint.split('-')[-1]) if train_manager.latest_checkpoint is not None else 0
except:
    train_baseEpoch = 0;

print('\n\n\n\n\n\n')
print('Loaded::', train_baseEpoch)
print('\n\n\n\n\n\n')



if __name__=='__main__':
    savefigs()
    ''' Main-training  '''
    print('Main Training...')
    ll_weight = tf.Variable(1.0)
    kl_weight = tf.Variable((5*(0+train_baseEpoch))/sim_epochs_ll_kl if ((5*(0+train_baseEpoch))/sim_epochs_ll_kl)<1.0 else 1.0)
    ''' Model Definition'''
    h_in = tfk.layers.Input([genData.T,genData.Kx,genData.Ky])
    g_t_1 = g_0 = 1e-2*tf.ones(shape=[sim_batch_size,dim_g]);
    obj_list = [] 
    kl_list = []
    loglike_list = []
    mse_list = []
    mean_h_t_list= []
    t=0
    for t in range(genData.T):
        ''' Step 2.1 q(z_t | g_t_1, Phi_h(h_t)) = Phi_enc(tf.concat([ho_t,g_t_1],axis=-1)): 
            Infer Latent Variable using True Sample, h_t, from dataset and Gist, g_t_1'''    
        ho_t = Phi_h(h_in[:,t])
        mean_z_t_inf, logvar_z_t_inf = Phi_enc(tf.concat([ho_t,g_t_1],axis=-1))
        z_t_inf = commons.reparameterize(mean_z_t_inf, logvar_z_t_inf)
        ''' Step 1.1 p(z_t | g_t_1) = Phi_prior(g_t_1): 
            Prior of Latent variable '''
        mean_z_t_prior, logvar_z_t_prior = Phi_prior(g_t_1)
        ''' Step 1.2 p(h_t | Phi_z(z_t), g_t_1) = Phi_dec(tf.concat([zo_t,g_t_1_dropped], axis=-1)): Generate Sample using Latent Variable and dropped out Gist '''
        g_t_1_dropped = g_t_1
        zo_t = z_t_inf
        mean_h_t, logvar_h_t = Phi_dec(tf.concat([zo_t,g_t_1_dropped], axis=-1))
        ''' Step 1.3 g_t = f_theta(g_t_1, Phi_h(h_t), Phi_z(z_t)): Generate Gist using True Sample and Reparameterised'''
        g_t = f_theta(tf.concat([ho_t,zo_t,g_t_1], axis=-1))
        ''' Step 2.1 Compute KL(q(z_t | g_t_1, Phi_h(h_t)) || p(z_t | g_t_1))'''
        kl = 0.5*( tf.math.reduce_sum(logvar_z_t_prior-logvar_z_t_inf, axis=-1)\
            - dim_z\
                + tf.reduce_sum(tf.exp(logvar_z_t_inf-logvar_z_t_prior),axis=-1) \
                    + tf.reduce_sum(tf.multiply(tf.multiply(tf.exp(-logvar_z_t_prior), (mean_z_t_prior-mean_z_t_inf)), (mean_z_t_prior-mean_z_t_inf)), axis=-1))
        ''' Step 2.2 Compute log p(h_t | Phi_z(z_t), g_t_1)'''
        _arg1 = tf.reshape(h_in[:,t], [sim_batch_size,genData.Kx*genData.Ky])
        _arg2 = tf.reshape(mean_h_t, [sim_batch_size,genData.Kx*genData.Ky])
        _arg3 = tf.reshape(logvar_h_t, [sim_batch_size,genData.Kx*genData.Ky])
        mse = tf.reduce_sum(tf.square(tf.abs(_arg1-_arg2)), axis=-1)
        loglike = commons.logProbNormalUncorrelated(_arg1,_arg2,_arg3)
        loglike = tf.reduce_sum(loglike,axis=-1)
        obj_t = loglike - kl
        kl_list.append(kl)
        loglike_list.append(loglike)
        obj_list.append(obj_t)
        mse_list.append(mse)
        mean_h_t_list.append(mean_h_t)
        g_t_1 = g_t


    mse_list_loss = tf.reduce_mean(mse_list, axis=0)
    loglike_loss = -tf.reduce_mean(loglike_list, axis=0)
    kl_loss = tf.reduce_mean(kl_list, axis=0)
    obj_loss = -tf.reduce_mean(obj_list, axis=0)
    h_out = tf.stack(mean_h_t_list, axis=1)
    train_model_debug = tfk.Model(inputs=h_in, outputs=[loglike_loss, kl_loss, obj_loss, mse_list_loss, h_out])
    train_model = tfk.Model(inputs=h_in, outputs=[loglike_loss, kl_loss])

    ''' Training '''
    _gen = dataGenerator(_batch_size=sim_batch_size)
    _self_loss=lambda y_true, y_pred: y_pred
    cb = tf.keras.callbacks.Callback()
    def onEpochEnd(epoch, logs=None):
        train_manager.save();
        savefigs(epoch+train_baseEpoch)
        kl_weight.assign((5*(epoch+train_baseEpoch))/sim_epochs_ll_kl if ((5*(epoch+train_baseEpoch))/sim_epochs_ll_kl)<1.0 else 1.0)


    def onEpochEnd_noKL(epoch, logs=None):
        train_manager.save();
        savefigs(epoch+train_baseEpoch)
        # kl_weight.assign((5*(epoch+train_baseEpoch))/sim_epochs_ll_kl if ((5*(epoch+train_baseEpoch))/sim_epochs_ll_kl)<1.0 else 1.0)


    # h_in = next(_gen_main)[0][0]
    # [loglike_loss, kl_loss, obj_loss, mse_list_loss, h_out] = train_model_debug(h_in)
    _gen_main = dataGenerator(_batch_size=sim_batch_size, num=len(train_model.outputs))
    print('Training main KL... ')
    cb.on_epoch_end = onEpochEnd_noKL
    pre_train_model.trainable=True;
    if sim_reparam_in_pretrain:
        Phi_enc.trainable=False;
        Phi_dec.trainable=False;
    else:
        Phi_enc_mean.trainable=False;
        Phi_dec_mean.trainable=False;
    f_theta.trainable=False;
    # train_model.trainable=False;
    print('Trainable variables:',np.sum([np.prod(v.get_shape().as_list()) for v in train_model.trainable_variables]), np.sum([np.prod(v.get_shape().as_list()) for v in train_model.variables]))
    train_model.compile(loss=_self_loss,optimizer=train_opt,loss_weights=[1.0, 1.0])
    train_model.fit(_gen_main, epochs=sim_epochs_kl-(train_baseEpoch), steps_per_epoch=sim_steps_per_epoch, callbacks=[cb])
    print('Training main LL+KL... ')
    cb.on_epoch_end = onEpochEnd
    train_model.trainable=False; train_model.trainable=True
    f_theta.trainable=True;
    print('Trainable variables:',np.sum([np.prod(v.get_shape().as_list()) for v in train_model.trainable_variables]), np.sum([np.prod(v.get_shape().as_list()) for v in train_model.variables]))
    train_model.compile(loss=_self_loss,optimizer=train_opt,loss_weights=[1.0, kl_weight])
    train_model.fit(_gen_main, epochs=sim_epochs_ll_kl-(train_baseEpoch-sim_epochs_kl), steps_per_epoch=sim_steps_per_epoch, callbacks=[cb])
    # 100/100 [==============================] - 38s 374ms/step - loss: 2297.9358 - tf.math.negative_9_loss: 2297.8540 - tf.math.reduce_mean_2_loss: 0.0817
    print('Fine Tuning... ')
    cb.on_epoch_end = onEpochEnd_noKL
    print('Trainable variables:',np.sum([np.prod(v.get_shape().as_list()) for v in train_model.trainable_variables]), np.sum([np.prod(v.get_shape().as_list()) for v in train_model.variables]))
    train_model.compile(loss=_self_loss,optimizer=train_opt_finetune,loss_weights=[1.0, 1.0])
    train_model.fit(_gen_main, epochs=sim_epochs_finetuning-(train_baseEpoch-sim_epochs_ll_kl-sim_epochs_kl), steps_per_epoch=sim_steps_per_epoch, callbacks=[cb])
    pre_train_model.fit(_gen, epochs=1, steps_per_epoch=sim_steps_per_epoch, callbacks=[cb_pre])

'''
np.sum([np.prod(v.get_shape().as_list()) for v in train_model.trainable_variables])
pre_train_model.trainable=False
np.sum([np.prod(v.get_shape().as_list()) for v in train_model.trainable_variables])
pre_train_model.trainable_variables
pre_train_model.fit(_gen, epochs=1, steps_per_epoch=10)
'''
# t = next(_gen)
# p = train_model_debug(t[0])[-1]
# p = np.array(p); t = np.array(t)
# plt.figure(); plt.imshow(p[0,0]);
# plt.figure(); plt.imshow(p[0,0]); plt.show()

# exit()







# Error_Counter = 0
# def checkGrads(_loss, _gradients):
#     global Error_Counter
#     _gradients_check1 = [np.any(np.isinf(_ if _ is not None else 1)) for _ in _gradients]
#     _gradients_check2 = [np.any(np.isnan(_ if _ is not None else 1)) for _ in _gradients]
#     if np.isinf(_loss)|np.isnan(_loss)|np.any(np.isinf(_gradients_check1))|np.any(np.isnan(_gradients_check2)):
#         print('#',end='',flush=True)
#         Error_Counter= Error_Counter+1
#         if Error_Counter>10:
#             exit(); # we expect a run manager to re-run the program
#         if train_debug>5:
#             print(np.isinf(_loss))
#             print(np.isnan(_loss))
#             print(np.any(np.isinf(_gradients_check1)))
#             print(np.any(np.isnan(_gradients_check2)))
#         return None;
#     else:
#         return _gradients

# def EndOfEpoch(pretrain=False):
#     if not pretrain:
#         if (_epoch+1)%sim_decay_epoch_lr == 0:
#             print('!',end='',flush=True)
#             if train_opt.lr > sim_min_lr:
#                 train_opt.lr = (train_opt.lr*(10**-sim_decay_lr))
#             else:
#                 train_opt.lr = sim_min_lr
#     train_manager.save();
#     savefigs();
#     if not pretrain:
#         print('Model Saved, OBJ :', logg_epochs_obj[-1])
#         print('LL  :', logg_epochs_ll[-1], ', scale :',scale_ll)
#         print('KL  :', logg_epochs_kl[-1], ', scale :',scale_kl)
#         print('MSE :', logg_epochs_mse[-1])
#         print('dropout_rate_gist :', dropout_rate_gist)
#         print('lr :', train_opt.lr.numpy(), ', epoch :', str(_epoch)+'/'+str(sim_epochs_ll_kl))




# logg_epochs_ll = [];
# logg_epochs_mse = [];
# logg_epochs_kl = [];
# logg_epochs_obj = [];
# logg_in_epochs_ll = [];
# logg_in_epochs_mse = [];
# logg_in_epochs_kl = [];
# logg_in_epochs_obj = [];

# import time
# if __name__ == '__main__':
#     _epoch = -1; 
#     if train_baseEpoch==0:
#         savefigs();
#     __epoch=0
#     for __epoch in range(sim_epochs_ll_kl):
#         print('@',end='',flush=True)
#         _epoch = __epoch + train_baseEpoch
#         __time = time.time()
#         scale_kl = (1-powpow(-sim_decay_kl*(sim_base_epoch_kl+_epoch))).astype(ftype)
#         scale_ll = 1.0#(1-powpow(-sim_decay_ll*(sim_base_epoch_ll+_epoch))).astype(ftype)
#         dropout_rate_gist = 0;#powpow(-sim_decay_rate*(sim_base_epoch_rate+_epoch)).astype(ftype)
#         # channels_s = myutils.load(('channels_s__' if sim_LOS else 'channels_NLOS_s__')+str((5+_file)%train_files), folderName='data', subFolderName='data'+str(sim_loadDataConfig))
#         for _batch_idx in range(sim_batches_per_epoch):
#             _time = time.time()
#             _gotThrough = False; #doOnce();
#             while not _gotThrough:
#                 _channel_samples = genData.sample(numSamples=sim_batch_size)# np.abs(channels_s[train_idxs[_batch_idx]:train_idxs[_batch_idx+1]])
#                 with tf.GradientTape() as tape:
#                     try:
#                         objective, metadata = ForwardLosses(_channel_samples, scale_kl=scale_kl, scale_ll=scale_ll, dropout_rate_gist=dropout_rate_gist, \
#                             onlyMSE=False)#_epoch<10)
#                     except:
#                         print('wtf!', flush=True, end='');
#                     loss = -objective
#                 # print(np.mean(objective))
#                 gradients = tape.gradient(loss, trainable_variables)
#                 gradients = checkGrads(loss, gradients);
#                 _gotThrough = gradients is not None
#                 # if not _gotThrough and train_debug>0:
#                 #     print('\n',metadata[1].numpy(), metadata[2].numpy)
#             train_opt.apply_gradients(
#                 (grad, var) 
#                 for (grad, var) in zip(gradients, trainable_variables) 
#                 if grad is not None
#             )
#             logg_in_epochs_ll.append(np.mean(metadata[1]))
#             logg_in_epochs_kl.append(np.mean(metadata[2]))
#             logg_in_epochs_mse.append(np.mean(metadata[3]))
#             logg_in_epochs_obj.append(np.mean(objective))
#             if (_batch_idx+1)%(sim_batches_per_epoch//20)==0:
#                 print('.',end='',flush=True)
#         print()
#         logg_epochs_ll.append(np.mean(logg_in_epochs_ll))
#         logg_epochs_kl.append(np.mean(logg_in_epochs_kl))
#         logg_epochs_mse.append(np.mean(logg_in_epochs_mse))
#         logg_epochs_obj.append(np.mean(logg_in_epochs_obj))
#         logg_in_epochs_ll = []
#         logg_in_epochs_kl = []
#         logg_in_epochs_mse = []
#         logg_in_epochs_obj = []
#         EndOfEpoch()

