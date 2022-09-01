#!pip install tensorflow-gpu
#!pip install Keras

#!pip install -q pyyaml h5py  # Required to save models in HDF5 format
import tensorflow as tf
#%load_ext tensorboard
import datetime, os
print(tf.__version__)

#!pip install -U featuretools 
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import glob,os
from tensorflow.keras.layers import Dense,GRU, Flatten, Conv2D, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (Callback, TensorBoard, EarlyStopping,
                             ModelCheckpoint, CSVLogger, ProgbarLogger)
#from tensorflow.keras.layers import (Input, Dense, TimeDistributed, LSTM, GRU, Dropout, merge, 
 #                        Concatenate, Flatten, RepeatVector, Lambda, Bidirectional, SimpleRNN)
from tensorflow.keras.layers import (Input, Dense, TimeDistributed, LSTM, GRU, Dropout,Concatenate, Flatten, RepeatVector, Lambda, Bidirectional, SimpleRNN)
import sys
import csv
from collections import Iterable, OrderedDict
import datetime
from sklearn.manifold import TSNE
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from tensorflow.keras.layers import SimpleRNN,Bidirectional,BatchNormalization
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.losses import KLDivergence,binary_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler

lcs_file ='lcs.npy'
lcs =np.load(lcs_file,allow_pickle=True)
print(lcs[100][0:5])

#shuffling the data to avoid systematics 
np.random.seed(42)
np.random.shuffle(lcs)
lcs[100][0:5]

lcs_raw = pad_sequences(lcs, value=np.nan, dtype='float', padding='post')
print(lcs_raw[100][0:5])
#print(lcs_raw[:,:,1])
#print(np.nanmax(lcs_raw[:, :, 1], axis=1)) this is magnitude value
#print(np.inf)
#print(np.all(np.isnan(lcs_raw[:, :, 1])) | (np.nanmax(lcs_raw[:, :, 1], axis=1) > np.inf))

#functions taken from https://github.com/yutarotachibana/CatalinaQSO_AutoEncoder

def times_to_lags(T):
    """(N x n_step) matrix of times -> (N x n_step) matrix of lags.
    First time is assumed to be zero.
    """
    assert T.ndim == 2, "T must be an (N x n_step) matrix"
    return np.c_[np.diff(T, axis=1)/365., np.zeros(T.shape[0])]


def preprocess(X_raw, m_max=np.inf):
    X = X_raw.copy()
    #print(X)
    wrong_units =  np.all(np.isnan(X[:, :, 1])) | (np.nanmax(X[:, :, 1], axis=1) > m_max)
    #print(wrong_units)
    X = X[~wrong_units, :, :]
    X[:, :, 0] = times_to_lags(X[:, :, 0])
    #print(X[:, :, 0])
    means = np.atleast_2d(np.nanmean(X[:, :, 1], axis=1)).T
    #print(means)
    X[:, :, 1] -= means
    scales = np.atleast_2d(np.nanstd(X[:, :, 1], axis=1)).T
    #print(scales)
    X[:, :, 1] /= scales
    errors = X[:, :, 2] / scales
    #print(errors)
    X = X[:, :, :2]
    #print(X) is lcs_scaled value
    return X, means, scales, errors, wrong_units
    
lcs_scaled, means, scales, errors, wrong_units = preprocess(lcs_raw)

print(lcs_scaled[100][0:5])
print(means[100])
#print(lcs_scaled[1])
main_input = Input(shape=(lcs_scaled.shape[1], 2), name='main_input') #(lag, mag)
aux_input = Input(shape=(lcs_scaled.shape[1], 1), name='aux_input') #(lag)
#print(aux_input.shape)

model_input = [main_input, aux_input]

sample_weight = 1. / errors
#print(sample_weight)
sample_weight[np.isnan(sample_weight)] = 0.0
lcs_scaled[np.isnan(lcs_scaled)] = 0.

lcs_scaled[:, :, [1]]

#Initialization 
lr = 1e-3 #learning rate 
optimizer = Adam(lr=lr)
#output_size=16
output_size =2
#gru_size = 32
gru_size =10
#nepochs = 2000
nepochs =1000
#batchsize = 512
batchsize =100
#dropout_val = 0.25
dropout_val = 0.20

resume_training = False # if True use W&B to recover weights and resume training, if False train a new model

def sampling(samp_args):
    z_mean, z_log_sigma = samp_args

    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_sigma) * epsilon
    
    
#encoder
encoder = Bidirectional(GRU(gru_size, name='encoder1', return_sequences=True))(main_input)
encoder = Dropout(dropout_val, name='drop_encoder1')(encoder) 
encoder = Bidirectional(GRU(gru_size, name='encoder2', return_sequences=False))(encoder)
encoder = Dropout(dropout_val, name='drop_encoder2')(encoder)
codings_mean = Dense(units=output_size, name='encoding_mean', activation='linear')(encoder)
codings_log_var = Dense(units=output_size, name='encoding_log_var', activation='linear')(encoder) 
codings = Lambda(sampling, output_shape=(output_size,))([codings_mean, codings_log_var])


#decoder
decoder = RepeatVector(lcs_scaled.shape[1], name='repeat')(codings)
#decoder = tf.keras.layers.merge.concatenate([aux_input, decoder])
decoder = tf.keras.layers.concatenate([aux_input, decoder])
decoder = GRU(gru_size, name='decoder1', return_sequences=True)(decoder)
decoder = Dropout(dropout_val, name='drop_decoder1')(decoder)
decoder = GRU(gru_size, name='decoder2', return_sequences=True)(decoder)
decoder = TimeDistributed(Dense(1, activation='linear'), name='time_dist')(decoder)

#VAE
model = Model(model_input, decoder)

latent_loss = -0.5*K.sum(1+codings_log_var-K.exp(codings_log_var)-K.square(codings_mean),axis=-1)
model.add_loss(K.mean(latent_loss)/200.)
model.compile(optimizer=optimizer, loss='mse',  metrics=[tf.keras.metrics.MeanAbsoluteError()], weighted_metrics=[tf.keras.metrics.MeanAbsoluteError()], sample_weight_mode='temporal')#,run_eagerly=True)

model.summary()

log_dir = './lcs_image/'
weights_path = os.path.join(log_dir, 'weights_lr1e3_4paper.h5')
logs = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

early_stopping = EarlyStopping(monitor='val_loss', patience=50, mode='auto', restore_best_weights=True)
print(early_stopping)
check_points = tf.keras.callbacks.ModelCheckpoint(filepath=weights_path, save_freq='epoch', save_weights_only=False, save_best_only=True, monitor='val_loss', verbose=1)


early_stopping = EarlyStopping(monitor='val_loss', patience=50, mode='auto', restore_best_weights=True)
#print(early)
check_points = tf.keras.callbacks.ModelCheckpoint(filepath=weights_path, save_freq='epoch', save_weights_only=False, save_best_only=True, monitor='val_loss', verbose=1)

#aux input contains the delta times, Y (lcs_scaled[:, :, [1]]) is the arrey with the normalized magnitudes, sample_weights is 1/err
history = model.fit({'main_input': lcs_scaled, 'aux_input': np.delete(lcs_scaled, 1, axis=2)},
                    lcs_scaled[:, :, [1]], 
                    epochs=nepochs, 
                    batch_size=batchsize,
                    sample_weight=sample_weight,
                    callbacks = [
                                check_points,
                                early_stopping
                                 ],
                    validation_split=0.2,
                    )
                    
                  
train_loss = history.history['loss']
print('the train loss is',train_loss)
val_loss   = history.history['val_loss']
print('the validation loss is',val_loss)
train_mae  = history.history['mean_absolute_error']
val_mae    = history.history['val_mean_absolute_error']
train_wmae  = history.history['weighted_mean_absolute_error']
val_wmae    = history.history['val_weighted_mean_absolute_error']
xc         = range(len(train_loss))


df_history = pd.DataFrame({'epoch': xc, 'train_loss': train_loss, 'val_loss': val_loss, 
                           'train_mae': train_mae, 'val_mae': val_mae,
                           'train_wmae': train_wmae, 'val_wmae': val_wmae})

history_path = os.path.join(log_dir, 'train_history_4paper_lr1e3.csv')


df_history.to_csv(history_path)

#print(xc)
#data=pd.read_csv(history_path)
#print(data)
print(val_loss)

plt.figure()
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.savefig('./lcs_image/xc_val_loss_and_train_loss.png')

model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
plt.savefig('./lcs_image/GRU2x32-encoding16_loss.png')


encode_model = Model(model.input, model.get_layer('lambda').output)
decode_model = Model(model.input, model.output)
print(encode_model)
print(decode_model)
encoding = encode_model.predict({'main_input': lcs_scaled, 'aux_input': np.delete(lcs_scaled, 1, axis=2)})
print(encoding)
decoding = decode_model.predict({'main_input': lcs_scaled, 'aux_input': np.delete(lcs_scaled, 1, axis=2)})
print(decoding)


encoding_train = encoding[0:int(0.8*len(lcs))]
print(encoding_train)
encoding_val = encoding[int(0.8*len(lcs)):len(lcs)]
decoding_train = decoding[0:int(0.8*len(lcs))]
decoding_val = decoding[int(0.8*len(lcs)):len(lcs)]
X_raw_train = lcs_raw[0:int(0.8*len(lcs))]
X_raw_val = lcs_raw[int(0.8*len(lcs)):len(lcs)]
X_train = lcs_scaled[0:int(0.8*len(lcs))]
X_val = lcs_scaled[int(0.8*len(lcs)):len(lcs)]
scales_train = scales[0:int(0.8*len(lcs))]
scales_val = scales[int(0.8*len(lcs)):len(lcs)]
means_train = means[0:int(0.8*len(lcs))]
means_val = means[int(0.8*len(lcs)):len(lcs)]

i = 150 #init num of the plotting data
#i=15000
k=0 #initialize the counter
fignum = 5
plt.figure(figsize=(12, 4))
#plotting raw-lightcurves and the decoded lightcurves 
for num in range(i,i+fignum):
    plt.subplot(2, fignum, k+1)
    #print(X_raw_train[num][:,0])
    #print(365.0*np.cumsum(X_train[num][:, 0][X_train[num][:, 0]>0])+X_raw_train[num][0,0])
    plt.errorbar(X_raw_train[num][:,0], X_raw_train[num][:,1], yerr=X_raw_train[num][:, 2], fmt='o', color='black', ms=5)
    plt.errorbar(365.0*np.cumsum(X_train[num][:, 0][X_train[num][:, 0]>0])+X_raw_train[num][0,0], 
             decoding_train[num][X_train[num][:, 0]>0]*scales_train[num]+means_train[num], fmt='o', color='orange', ms=7)
    plt.ylim(plt.ylim()[::-1])
    k += 1
#plotting encoded features
k=0
for num in range(i+fignum,i+2*fignum):
    plt.subplot(2, fignum, k+1+fignum)
    plt.imshow(encoding_train[num].reshape(4,4), vmin=-2, vmax=2, cmap='viridis_r')
    k += 1
plt.tight_layout()
plt.savefig('./lcs_image/ZTF_alerts_modeled_lcs_training_set.png')
plt.show()

print("""Upper panels: Input light curves (black points) and decoded light curves (orange points) \n
Lower panels: Encoded 16 features""")

#i = 1650 #init num of the plotting data
i=16
k=0 #initialize the counter
fignum = 5
plt.figure(figsize=(12, 4))
#plotting raw-lightcurves and the decoded lightcurves 
for num in range(i,i+fignum):
    plt.subplot(2, fignum, k+1)
    plt.errorbar(X_raw_val[num][:,0], X_raw_val[num][:,1], yerr=X_raw_val[num][:, 2], fmt='o', color='black', ms=5)
    plt.errorbar(365.0*np.cumsum(X_val[num][:, 0][X_val[num][:, 0]>0])+X_raw_val[num][0,0], 
             decoding_val[num][X_val[num][:, 0]>0]*scales_val[num]+means_val[num], fmt='o', color='orange', ms=7)
    plt.ylim(plt.ylim()[::-1])
    k += 1
#plotting encoded features
k=0
for num in range(i+fignum,i+2*fignum):
    plt.subplot(2, fignum, k+1+fignum)
    plt.imshow(encoding_val[num].reshape(4,4), vmin=-2, vmax=2, cmap='viridis_r')
    k += 1
plt.tight_layout()
plt.savefig('./lcs_image/ZTF_alerts_modeled_lcs_validation_set.png')
plt.show()

def calc_redchisq(x, x_pred, weight):
    mask =  (~np.isnan(weight))
    out = np.sum(((x[mask]-x_pred[mask])*weight[mask])**2)/len(weight[mask])
    return out

RedChiSq = []
print('Caluculating reduced chi-square for each source')
for m in range(0, len(lcs_raw)):
    if m%100 == 0:
        print('.', end='')
    RedChiSq.append(calc_redchisq(lcs_scaled[m][:,1]*scales[m]+means[m], np.squeeze(decoding[m]*scales[m]+means[m]), 1/lcs_raw[m][:,2]))
    
RedChiSq_train = RedChiSq[0:int(0.8*len(lcs))]
RedChiSq_val = RedChiSq[int(0.8*len(lcs)):len(lcs)]    

print("training chi2 median, mean: ",np.median(np.array(RedChiSq_train)),np.mean(np.array(RedChiSq_train)))
print("validation chi2 median, mean: ",np.median(np.array(RedChiSq_val)),np.mean(np.array(RedChiSq_val)))

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.hist(np.array(RedChiSq_train)[~np.isnan(RedChiSq_train)], 
         bins=10**np.arange(-1, 3, 0.1))
plt.xscale('log')
plt.xlabel(r'$\chi_{{\rm red}}^2$', fontsize=15)
plt.ylabel('Number of sources', fontsize=15)
plt.title('Training dataset', fontsize=15)
plt.tight_layout()

plt.subplot(1,2,2)
plt.hist(np.array(RedChiSq_val)[~np.isnan(RedChiSq_val)], 
         bins=10**np.arange(-1, 3, 0.1))
plt.xscale('log')
plt.xlabel(r'$\chi_{{\rm red}}^2$', fontsize=15)
plt.ylabel('Number of sources', fontsize=15)
plt.title('Validation dataset', fontsize=15)
plt.tight_layout()

plt.savefig('./lcs_image/ZTF_alerts_chisquare_distribution.png')
plt.show()


X_reduced_train =TSNE(n_components=2, perplexity=100, random_state=32, n_iter=1000).fit_transform(encoding_train)








