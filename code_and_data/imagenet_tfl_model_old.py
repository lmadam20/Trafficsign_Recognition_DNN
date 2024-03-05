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

# %%
# load the datasets
path = './datasets/GTSRB/'

print('Loading datasets...')
training = np.load(os.path.join(path, 'training.npz'), allow_pickle=True)
testing = np.load(os.path.join(path, 'testing.npz'), allow_pickle=True)

X_train = training['X']
y_train = training['y']

X_test = testing['X']
y_test = testing['y']

# %%
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

# %%
# resizing
print('Resizing...')

X_train = tf.image.resize(X_train, [224, 224], method='nearest')
X_valid = tf.image.resize(X_valid, [224, 224], method='nearest')
X_test = tf.image.resize(X_test, [224, 224], method='nearest')

# %%
# preprocess for MobileNet
# print('Preprocessing for MobileNet...')

# X_train = tf.keras.applications.mobilenet.preprocess_input(X_train)
# X_valid = tf.keras.applications.mobilenet.preprocess_input(X_valid)
# X_test = tf.keras.applications.mobilenet.preprocess_input(X_test)

# %%
# build model
print('Searching and building a model based on MobileNet...')

def build_model(hp):
    activation_functions = ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh',
                            'selu', 'elu', 'exponential', 'leaky_relu', 'relu6', 'silu',
                            'gelu', 'hard_sigmoid', 'log_softmax', 'mish', 'linear']
    units_lc0 = hp.Int('units_lc0', 64, 256, 32)
    activation_lc0 = hp.Choice('activation_lc0', activation_functions)
    units_lc1 = hp.Int('units_lc1', 64, 256, 32)
    activation_lc1 = hp.Choice('activation_lc1', activation_functions)
    dropout_l = hp.Float('dropout_l', 0.0, 0.5)
    activation_final = hp.Choice('activation_final', activation_functions)

    mobilenet_model = tf.keras.applications.MobileNet(
        weights="imagenet",
        input_shape=(224, 224, 3),
        include_top=False
    )

    mobilenet_model.trainable = False

    input_layer = tf.keras.layers.Input(shape=(224, 224, 3))
    input_layer = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1)(input_layer)

    final_model = mobilenet_model(input_layer,
                                  training=False)
    final_model = tf.keras.layers.GlobalMaxPooling2D()(final_model)
    final_model = tf.keras.layers.Flatten()(final_model)
    final_model = tf.keras.layers.Dense(units_lc0, activation=activation_lc0)(final_model)
    final_model = tf.keras.layers.Dense(units_lc1, activation=activation_lc1)(final_model)
    final_model = tf.keras.layers.Dropout(dropout_l)(final_model)
    final_model = tf.keras.layers.Dense(43, activation=activation_final)(final_model) # 43 classes

    final_model = tf.keras.Model(input_layer, final_model)

    loss_from_logits = True
    if activation_final in ['sigmoid', 'softmax']:
        loss_from_logits = False

    final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=loss_from_logits),
                  metrics=['accuracy'])
    # final_model.summary()
    
    return final_model

tuner = kt.BayesianOptimization(build_model,
                        project_name='kt_imagenet_tfl_model',
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

# %%
# fit model
print('Fitting model...')
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_valid, y_valid), callbacks=[
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                     restore_best_weights=True)
])

# %%
# fine tuning
print('Fine-tuning model...')

current_weights = model.get_weights()

model.layers[2].trainable = True
loss_from_logits = True

if model.layers[-1].activation.__name__ in ['sigmoid', 'softmax']:
    loss_from_logits = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=loss_from_logits),
              metrics=['accuracy'])
model.set_weights(current_weights)
model.summary()

model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid), callbacks=[
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2,
                                     restore_best_weights=True)
])

# %%
# save model
model.save('mobilenet_tfl_model.keras')

# %%
# load model
model = tf.keras.models.load_model('mobilenet_tfl_model.keras')

# %%
# testing
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('Test accuracy:', test_acc)

# %%
