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