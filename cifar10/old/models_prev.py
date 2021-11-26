import numpy as np
import tensorflow.compat.v2 as tf
from sklearn.model_selection import train_test_split
import time
from datetime import datetime

tf.enable_v2_behavior()

from pllay import *

"""# Training"""

nmax_diag = 64


class MNIST_MLP(tf.keras.Model):
    def __init__(self, name='mnistmlp', unitsDense=64, **kwargs):
        super(MNIST_MLP, self).__init__(name=name, **kwargs)
        self.layer2 = tf.keras.layers.Dense(unitsDense, activation='relu', name='dense_2') 
        self.layer3 = tf.keras.layers.Dense(10, name='predictions')

    def call(self, x):
        xg, xl1, xl2, xd = tf.split(x, [784, 100, 162, 8*nmax_diag], axis=-1)
        x = xg
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class MNIST_MLP_PLLay(tf.keras.Model):
    def __init__(self, name='mnistmlppllay', unitsDense=64, unitsTop=32, **kwargs):
        super(MNIST_MLP_PLLay, self).__init__(name=name, **kwargs)
        self.layer1_1 = GThetaLayer(unitsTop)
        self.layer1_2 = GThetaLayer(unitsTop)
        self.layer2 = tf.keras.layers.Dense(unitsDense, activation='relu', name='dense_2') 
        self.layer3 = tf.keras.layers.Dense(10, name='predictions')

    def call(self, x):
        xg, xl1, xl2, xd = tf.split(x, [784, 100, 162, 8*nmax_diag], axis=-1)
        xl1 = tf.nn.relu(self.layer1_1(xl1))
        xl2 = tf.nn.relu(self.layer1_2(xl2))
        x = tf.concat((xg, xl1, xl2), -1)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class CIFAR10_CNN(tf.keras.Model):
    def __init__(self, name='cifar10cnn', filters=32, kernel_size=3, unitsDense=64, **kwargs):
        super(CIFAR10_CNN, self).__init__(name=name, **kwargs)
        self.layer1_1 = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", activation='relu')
        self.layer1_2 = tf.keras.layers.Conv2D(1, kernel_size, padding="same", activation='relu')
        self.layer3 = tf.keras.layers.Dense(unitsDense, activation='relu', name='dense_2')
        self.layer4 = tf.keras.layers.Dense(10, name='predictions')

    def call(self, x):
        xg, xl1, xl2, xd = tf.split(x, [3072, 100, 162, 8*nmax_diag], axis=-1)
        xg = tf.reshape(xg, [8, 32, 32, 3])
        xg1 = self.layer1_1(xg)
        xg1 = self.layer1_2(xg1)
        xg1 = tf.reshape(xg1, [8, 1024])
        x = xg1
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class MNIST_CNN_PLLay_Input(tf.keras.Model):
    def __init__(self, name='mnistcnnpllayinput', filters=32, kernel_size=3, unitsDense=64, unitsTopInput=32, **kwargs):
        super(MNIST_CNN_PLLay_Input, self).__init__(name=name, **kwargs)
        self.layer1_1 = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", activation='relu')
        self.layer1_2 = tf.keras.layers.Conv2D(1, kernel_size, padding="same", activation='relu')
        self.layer2_1 = GThetaLayer(unitsTopInput)
        self.layer2_2 = GThetaLayer(unitsTopInput)
        self.layer3 = tf.keras.layers.Dense(unitsDense, activation='relu', name='dense_2')
        self.layer4 = tf.keras.layers.Dense(10, name='predictions')

    def call(self, x):
        xg, xl1, xl2, xd = tf.split(x, [784, 100, 162, 8*nmax_diag], axis=-1)
        xg = tf.reshape(xg, [16, 28, 28, 1])
        xg1 = self.layer1_1(xg)
        xg1 = self.layer1_2(xg1)
        xg1 = tf.reshape(xg1, [16, 784])
        xl1 = tf.nn.relu(self.layer2_1(xl1))
        xl2 = tf.nn.relu(self.layer2_2(xl2))
        x = tf.concat((xg1, xl1, xl2), -1)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class CIFAR10_CNN_PLLay(tf.keras.Model):
    def __init__(self, name='cifar10cnnpllay', filters=32, kernel_size=3, unitsDense=64, unitsTopInput=32, unitsTopMiddle=64, **kwargs):
        super(CIFAR10_CNN_PLLay, self).__init__(name=name, **kwargs)
        self.layer1_1 = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", activation='relu')
        self.layer1_2 = tf.keras.layers.Conv2D(1, kernel_size, padding="same", activation='relu')
        self.layer1_3 = TopoFunLayer(unitsTopMiddle, grid_size=[32, 32], tseq=np.linspace(0.05, 0.95, 18), KK=list(range(3)))
        self.layer2_1 = GThetaLayer(unitsTopInput)
        self.layer2_2 = GThetaLayer(unitsTopInput)
        self.layer3 = tf.keras.layers.Dense(unitsDense, activation='relu', name='dense_2')
        # self.dp = tf.keras.layers.Dropout(0.2) 
        self.layer4 = tf.keras.layers.Dense(10, name='predictions')

    def call(self, x):
        xg, xl1, xl2, xd = tf.split(x, [3072, 100, 162, 8*nmax_diag], axis=-1)
        xg = tf.reshape(xg, [8, 32, 32, 3])
        xg1 = self.layer1_1(xg)
        xg1 = self.layer1_2(xg1)
        xg1 = tf.reshape(xg1, [8, 1024])
        xg1_1 = tf.nn.relu(self.layer1_3(xg1))
        xg1 = tf.concat((xg1, xg1_1), -1)
        xl1 = tf.nn.relu(self.layer2_1(xl1))
        xl2 = tf.nn.relu(self.layer2_2(xl2))
        x = tf.concat((xg1, xl1, xl2), -1)
        x = self.layer3(x)
        # x = self.dp(x)
        x = self.layer4(x)
        return x


class MNIST_CNN_PLLay_CnnTakesDtm(tf.keras.Model):
    def __init__(self, name='mnistcnnpllaycnntakesdtm', filters=32, kernel_size=3, unitsDense=64, unitsTopInput=16, unitsTopMiddle=16, **kwargs):
        super(MNIST_CNN_PLLay_CnnTakesDtm, self).__init__(name=name, **kwargs)
        self.layer1_1 = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", activation='relu')
        self.layer1_2 = tf.keras.layers.Conv2D(1, kernel_size, padding="same", activation='relu')
        self.layer2_1 = DTMWeightWrapperLayer(m0=0.05, by=1./13.5) 
        self.layer2_2 = DTMWeightWrapperLayer(m0=0.2, by=1./13.5) 
        self.layer3_1 = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", activation='relu')
        self.layer3_2 = tf.keras.layers.Conv2D(1, kernel_size, padding="same", activation='relu')
        self.layer4_1 = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", activation='relu')
        self.layer4_2 = tf.keras.layers.Conv2D(1, kernel_size, padding="same", activation='relu')
        self.layer3_3 = TopoFunLayer(unitsTopMiddle, grid_size=[28, 28], tseq=np.linspace(0.06, 0.3, 25), KK=list(range(2)))
        self.layer4_3 = TopoFunLayer(unitsTopMiddle, grid_size=[28, 28], tseq=np.linspace(0.14, 0.4, 27), KK=list(range(3)))
        self.layer5 = TopoFunLayer(unitsTopInput, grid_size=[28, 28], tseq=np.linspace(0.06, 0.3, 25), KK=list(range(2)))
        self.layer6 = TopoFunLayer(unitsTopInput, grid_size=[28, 28], tseq=np.linspace(0.14, 0.4, 27), KK=list(range(3)))
        self.layer7 = tf.keras.layers.Dense(unitsDense, activation='relu', name='dense_2') 
        self.layer8 = tf.keras.layers.Dense(10, name='predictions')

    def call(self, x):
        x1 = self.layer1_1(x)
        x1 = self.layer1_2(x1)
        x1 = tf.reshape(x1, [16, 784])
        x2 = tf.reshape(x, [16, 784])
        x2_1 = self.layer2_1(x2)
        x2_2 = self.layer2_2(x2)
        x3 = tf.reshape(x2_1, [16, 28, 28, 1])
        x3 = self.layer3_1(x3)
        x3 = self.layer3_2(x3)
        x3 = tf.reshape(x3, [16, 784])
        x3 = tf.nn.relu(self.layer3_3(x3))
        x4 = tf.reshape(x2_2, [16, 28, 28, 1])
        x4 = self.layer4_1(x4)
        x4 = self.layer4_2(x4)
        x4 = tf.reshape(x4, [16, 784])
        x4 = tf.nn.relu(self.layer4_3(x4))
        x5 = tf.nn.relu(self.layer5(x2_1))
        x6 = tf.nn.relu(self.layer6(x2_2))
        x = tf.concat((x1, x3, x4, x5, x6), -1)
        x = self.layer7(x)
        x = self.layer8(x)
        return x