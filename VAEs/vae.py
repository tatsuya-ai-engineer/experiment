# GPUを使用する

from tensorflow import keras
print(keras.__version__)

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")


# データセット準備
import numpy as np
from keras.utils import np_utils
from keras.datasets import mnist, cifar10

def create_dataset(data_name):
    if data_name==mnist:
        (x_tr, y_tr), (x_te, y_te) = mnist.load_data()
        x_tr, x_te = x_tr.astype('float32')/255., x_te.astype('float32')/255.
        x_tr = x_tr.reshape([-1,28,28,1])
        x_te = x_te.reshape([-1,28,28,1])
        y_tr = np_utils.to_categorical(y_tr)
        y_te = np_utils.to_categorical(y_te)

    if data_name==cifar10:
        (x_tr, y_tr), (x_te, y_te) = cifar10.load_data()
        x_tr, x_te = x_tr.astype('float32')/255., x_te.astype('float32')/255.
        x_tr = x_tr.reshape([-1,32,32,3])
        x_te = x_te.reshape([-1,32,32,3])
        y_tr = np_utils.to_categorical(y_tr)
        y_te = np_utils.to_categorical(y_te)

    return x_tr, y_tr, x_te, y_te

x_tr, y_tr, x_te, y_te = create_dataset(mnist)
print(x_tr.shape,y_tr.shape)


# VAEのモデルを作成
import keras

from keras.layers import Lambda, Input, Dense, Dropout
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.models import Sequential
from keras.layers import MaxPooling2D

from keras import layers
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose

from keras import backend as K

input_shape = np.array([x_tr.shape[1], x_tr.shape[2], x_tr.shape[3]])
print(input_shape)
kernel_size = 3
latent_dim = 16
recon_filter = x_tr.shape[3]

# 特徴量のサンプリング
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# VAEのモデル
def set_vae(input_shape, kernel_size, latent_dim, recon_filter):
    inputs = Input(shape=input_shape, name='encoder_input')
    
    x = inputs
    x = Conv2D(filters=32,
                kernel_size=kernel_size,
                activation='relu',
                strides=2,
                padding='same')(x)
    x = Conv2D(filters=64,
                kernel_size=kernel_size,
                activation='relu',
                strides=2,
                padding='same')(x)
    # shape info needed to build decoder model
    shape = K.int_shape(x)
    # generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    
    #plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)
    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    x = Conv2DTranspose(filters=64,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=2,
                            padding='same')(x)
    x = Conv2DTranspose(filters=32,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=2,
                            padding='same')(x)
    
    outputs = Conv2DTranspose(filters=recon_filter,
                              kernel_size=kernel_size,
                              activation='sigmoid',
                              padding='same',
                              name='decoder_output')(x)
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    #plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)
    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')
    #models = (encoder, decoder)
    #data = (x_test, y_test)
    # VAE loss = mse_loss or xent_loss + kl_loss
    reconstruction_loss = binary_crossentropy(K.flatten(inputs),
                                            K.flatten(outputs))
    reconstruction_loss *= x_tr.shape[1] * x_tr.shape[2]
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
    vae.summary()


    return vae


vae = set_vae(input_shape, kernel_size, latent_dim, recon_filter)


# モデルへのデータ入力と学習
epoch = 50
batch_size = 64

result = vae.fit(x=x_tr, epochs=epoch, batch_size=batch_size)

# パラメータ保存
vae.save_weights("/home/ueda-tatsuya/Experiments/parameter/vae_para.hdf5")