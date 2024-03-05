#!/bin/python3
# SPDX-FileCopyrightText: Copyright 2024 Leon Maurice Adam
# SPDX-License-Identifier: BSD-3-Clause-Modification

#%%
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import keras_tuner as kt
import sklearn as skl
from sklearn.model_selection import train_test_split

print('Tensorflow version: ' + str(tf.__version__))
print('Keras version: ' + str(keras.__version__))
print('KerasTuner version: ' + str(kt.__version__))
print('Scikit-learn version: ' + str(skl.__version__))

#%%
# load the datasets
path = './datasets/GTSRB/'

print('Loading datasets...')
training = np.load(os.path.join(path, 'training.npz'), allow_pickle=True)
testing = np.load(os.path.join(path, 'testing.npz'), allow_pickle=True)

X_train = training['X']
y_train = training['y']

X_test = testing['X']
y_test = testing['y']

#%%
# converting datatypes
X_train = np.asarray(X_train).astype('float32')
X_test = np.asarray(X_test).astype('float32')

# %%
# split training data into training and validation set
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2,
                                                      shuffle=True)

# %%
# augment data

def augment_data(X):
    layers = [
        tf.keras.layers.RandomRotation(1/8), # 45 degrees, either cw/ccw
        tf.keras.layers.RandomTranslation(0.1, 0.1) # shift w/h by up to 10%
    ]

    X_augmented = []

    for x in X:
        for layer in layers:
            x = layer(x)
        
        X_augmented.append(x)
    
    return np.asarray(X_augmented).astype('float32')

print('Augmenting training and validation data...')
X_train = tf.py_function(func=augment_data, inp=[X_train], Tout=tf.float32)
X_valid = tf.py_function(func=augment_data, inp=[X_valid], Tout=tf.float32)

#%%
# build model
print('Searching and building a model...')

def build_model(hp):
    activation_functions = ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh',
                            'selu', 'elu', 'exponential', 'leaky_relu', 'relu6', 'silu',
                            'gelu', 'hard_sigmoid', 'log_softmax', 'mish', 'linear']
    filters_0 = hp.Int('filters_0', 64, 256, 32)
    filters_1 = hp.Int('filters_1', 128, 384, 32)
    units_lc = hp.Int('units_lc', np.min([filters_1 - 32, 96]), np.max([filters_1 + 32, 128]), 8)
    activation_lc = hp.Choice('activation_lc', activation_functions)
    dropout_l = hp.Float('dropout_l', 0.0, 0.5)
    activation_final = hp.Choice('activation_final', activation_functions)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(50, 50, 3)),
        tf.keras.layers.Conv2D(filters_0, kernel_size=(5, 5)),
        tf.keras.layers.MaxPooling2D(pool_size=(5, 3)),
        tf.keras.layers.Conv2D(filters_1, kernel_size=(5, 5)),
        tf.keras.layers.MaxPooling2D(pool_size=(5, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units_lc, activation=activation_lc),
        tf.keras.layers.Dropout(dropout_l),
        tf.keras.layers.Dense(43, activation=activation_final)
    ])

    loss_from_logits = True
    if activation_final in ['sigmoid', 'softmax']:
        loss_from_logits = False

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=loss_from_logits),
                  metrics=['accuracy'])
    return model

tuner = kt.BayesianOptimization(build_model,
                        project_name='kt_convolutional_model',
                        objective='val_accuracy',
                        max_trials=20,
                        seed=42
                        )
tuner.search(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid), callbacks=[
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
])
tuner.results_summary()
model = tuner.get_best_models()[0]

model.summary()

#%%
# fit model
print('Fitting model...')
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_valid, y_valid), callbacks=[
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                     restore_best_weights=True)
])

#%%
# save model
model.save('convolutional_model.keras')

#%%
# load model
model = tf.keras.models.load_model('convolutional_model.keras')

#%%
# testing
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('Test accuracy:', test_acc)

#%%
