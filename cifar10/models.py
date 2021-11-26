import numpy as np
import tensorflow.compat.v2 as tf
from sklearn.model_selection import train_test_split
import time
from datetime import datetime
tf.enable_v2_behavior()
from pllay import *

nmax_diag = 64

class MLP(tf.keras.Model):
    def __init__(self, name='standard_mlp', **kwargs):
        super(MLP, self).__init__(name=name, **kwargs)
        self.flatten = tf.keras.layers.Flatten()
        self.layer1 = tf.keras.layers.Dense(512, activation='relu', name='dense_1')
        self.layer2 = tf.keras.layers.Dense(128, activation='relu', name='dense_2')
        self.layer3 = tf.keras.layers.Dense(10, name='predictions')

    def call(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class MLP_PLLAY(tf.keras.Model):
    def __init__(self, name='mlp_pllay', grid_size = [32,32], units_pllay = 128, **kwargs):
        super(MLP_PLLAY, self).__init__(name=name, **kwargs)
        self.flatten = tf.keras.layers.Flatten()
        self.layer1 = tf.keras.layers.Dense(512, activation='relu', name='dense_2')
        self.pllay = TopoFunLayer(units_pllay, grid_size=grid_size, tseq=np.linspace(0.05, 0.95, 18), KK=list(range(3)))
        self.layer2 = tf.keras.layers.Dense(10, name='predictions')

    def call(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.pllay(x)
        x = self.layer2(x)
        return x

class CNN(tf.keras.Model):
    def __init__(self, name='cifar10cnn', **kwargs):
        super(CNN, self).__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(32, 3, padding="same", activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, 3, padding="same", activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(128, 3, padding="same", activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu', name='dense_1')
        self.fc2 = tf.keras.layers.Dense(10, name='predictions')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class CNN_PLLAY(tf.keras.Model):
    def __init__(self, name='mlp_pllay', grid_size = [32,32], units_pllay = 512, **kwargs):
        super(CNN_PLLAY, self).__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(32, 3, padding="same", activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(128, 3, padding="same", activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.pllay = TopoFunLayer(units_pllay, grid_size=grid_size, tseq=np.linspace(0.05, 0.95, 18), KK=list(range(3)))
        self.fc1 = tf.keras.layers.Dense(128, activation='relu', name='dense_1')
        self.fc2 = tf.keras.layers.Dense(10, name='predictions')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.pllay(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x