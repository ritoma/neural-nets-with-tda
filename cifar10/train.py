import numpy as np
import tensorflow.compat.v2 as tf
from sklearn.model_selection import train_test_split
import time
from datetime import datetime
import wandb
from wandb.keras import WandbCallback
from tensorflow.compat.v2.keras.optimizers import SGD, Nadam, Adam, RMSprop
from models import *

tf.enable_v2_behavior()

config_defaults = {
        'epochs': 20,
        'batch_size': 8,
        'learning_rate': 1e-3
}

wandb.login()
wandb.init(entity = "devanshgupta", project = "cifar_10_train_pllay_20000_test", config = config_defaults)
train_pllay()

config = wandb.config
optimizer = Adam(lr=config.learning_rate, beta_1=0.9, beta_2=0.999, clipnorm=1.0)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model = CNN_PLLAY()   # CNN, MLP_PLLAY, MLP

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
size_train = x_train.shape[0]
size_test = x_test.shape[0]

train_acc = tf.keras.metrics.Mean()
val_acc = tf.keras.metrics.Mean()

for epoch in range(1, config.epochs + 1):
    for batch_idx in range(0, size_train, config.batch_size):
        inputs = x_train[batch_idx:batch_idx+config.batch_size, :, :, :]
        targets = y_train[batch_idx:batch_idx+config.batch_size, :]
        with tf.GradientTape() as tape:
            logits = model(inputs, training=True)
            train_loss = loss_fn(targets, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        acc_value = tf.math.equal(targets, tf.math.round(tf.keras.activations.softmax(logits)))
        train_acc.update_state(acc_value)
        wandb.log({"Train_Loss": train_loss})

    # Validation loop
    for batch_idx in range(0, size_test, config.batch_size):
        inputs = x_test[batch_idx:batch_idx+config.batch_size, :, :, :]
        targets = y_test[batch_idx:batch_idx+config.batch_size, :]
        val_logits = model(inputs, training=False)
        loss_value = loss_fn(targets, val_logits)
        acc_value = tf.math.equal(targets, tf.math.round(tf.keras.activations.softmax(val_logits)))
        val_acc.update_state(acc_value)
        wandb.log({"Validation Loss": val_loss})

    wandb.log({"Train Accuracy": train_acc.result(),
               "Validation Accuracy": val_acc.result()})
    
    model.save_weights(ckpt_dir)