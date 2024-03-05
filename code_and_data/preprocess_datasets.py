#!/bin/python3
# SPDX-FileCopyrightText: Copyright 2024 Leon Maurice Adam
# SPDX-License-Identifier: BSD-3-Clause-Modification

#%%
import numpy as np
import skimage as ski
import pandas as pd
import matplotlib.pyplot as plt
import os

#%%
path = './datasets/GTSRB/'

train_path = 'training/'
test_path = 'test/'

full_train_path = os.path.join(path, train_path)
full_test_path = os.path.join(path, test_path)

RESIZING_DIM = 60
CROPPING_DIM = (50, 50)

# expect following directory structure
# - <path>/
# |--- <train_path>/
# |----|--- 1/
# |----|----|---- <image 0>.ppm
# |----|----|---- <image 1>.ppm
# |----|----|---- ...
# |----|----|---- <image m>.ppm
# |----|----|---- <annotations for 1>.csv
# |----|--- 2/
# |----|----|---- <image 0>.ppm
# |----|----|---- <image 1>.ppm
# |----|----|---- ...
# |----|----|---- <image m>.ppm
# |----|----|---- <annotations for 2>.csv
# |----|--- ...
# |----|--- n/
# |----|----|---- <image 0>.ppm
# |----|----|---- <image 1>.ppm
# |----|----|---- ...
# |----|----|---- <image m>.ppm
# |----|----|---- <annotations for n>.csv

#%%
def reject_small_images(X, y, min_pixels=1600):
    if len(X) != len(y):
        raise ValueError('length of X and y should be equal')
    
    X_out = []
    y_out = []

    for i in range(len(y)):
        x = X[i]
        x_shape = np.shape(x)
        
        if x_shape[0] * x_shape[1] >= min_pixels:
            X_out.append(x)
            y_out.append(y[i])
    
    return np.array(X_out, dtype=object), np.asarray(y_out)

def select_samples(X, y, rng=np.random.default_rng(42), least_count=None):
    i_argsort = np.argsort(y)
    X_sorted = X[i_argsort]
    y_sorted = y[i_argsort]

    labels, label_indices, label_counts = np.unique(y_sorted, return_counts=True, return_index=True)

    if least_count is None:
        least_count = int(np.ceil(np.median(label_counts)))
        print('least_count not set, calculated ' + str(least_count) + ' using median of counts')

    X_new = []
    y_new = []

    for i in range(len(label_counts)):
        i_max = np.argmax(label_counts)

        # print('i_max = ' + str(i_max) + ', label_counts[i_max] = '
        #        + str(label_counts[i_max]) + ', labels[i_max] = ' + str(labels[i_max]))
        
        if label_counts[i_max] > least_count:
            K = range(label_counts[i_max])
            for j in range(least_count):
                k = rng.choice(K, replace=False)
                X_new.append(X_sorted[label_indices[i_max] + k])
                y_new.append(y_sorted[label_indices[i_max] + k])
        else:
            for j in range(label_indices[i_max], label_indices[i_max] + label_counts[i_max]):
                X_new.append(X_sorted[j])
                y_new.append(y_sorted[j])

        labels = np.delete(labels, i_max)
        label_indices = np.delete(label_indices, i_max)
        label_counts = np.delete(label_counts, i_max)
    
    return np.array(X_new, dtype=object), np.asarray(y_new)

def resize_images(X, target_dim=RESIZING_DIM):
    X_resized = []
    processed_counter = 0

    for x in X:
        dims = np.shape(x)
        new_dim0 = 0
        new_dim1 = 0

        # resize by lower dimension
        if dims[0] < dims[1]:
            new_dim0 = target_dim
            # new_dim1 = int(dims[1] * (target_dim / dims[0]))
            # new_dim1 = np.max([int(new_dim0 * 0.9), new_dim1])
            new_dim1 = np.clip(int(dims[1] * (target_dim / dims[0])),
                                    int(new_dim0 * 0.9), int(new_dim0 * 1.1))
        else:
            new_dim1 = target_dim
            # new_dim0 = int(dims[0] * (target_dim / dims[1]))
            # new_dim0 = np.max([int(new_dim1 * 0.9), new_dim0])
            new_dim0 = np.clip(int(dims[0] * (target_dim / dims[1])),
                                    int(new_dim1 * 0.9), int(new_dim1 * 1.1))
        
        # print('old dims: ' + str(dims) + ', new dims: ' + str((new_dim0, new_dim1)))

        x_resized = ski.transform.resize(x, (new_dim0, new_dim1), anti_aliasing=True)
        # print('final dims: ' + str(np.shape(x_resized)))
        X_resized.append(x_resized)

        processed_counter += 1
        print(str(processed_counter) + ' images processed', end='\r')
    
    print(str(processed_counter) + ' images processed')
    return np.array(X_resized, dtype=object)

def centercrop_images(X, target_dim=CROPPING_DIM):
    X_cropped = []
    processed_counter = 0

    for x in X:
        dims = np.shape(x)
        dim0_diff = dims[0] - target_dim[0]
        dim1_diff = dims[1] - target_dim[1]

        crop_x0 = int(dim0_diff / 2)
        crop_x1 = crop_x0 + target_dim[0]
        crop_y0 = int(dim1_diff / 2)
        crop_y1 = crop_y0 + target_dim[1]

        x_cropped = x[crop_x0:crop_x1,crop_y0:crop_y1,:]
        X_cropped.append(x_cropped)

        processed_counter += 1
        print(str(processed_counter) + ' images processed', end='\r')
    
    print(str(processed_counter) + ' images processed')
    return np.array(X_cropped, dtype=object)

def equalize_images(X):
    X_eq = []
    processed_counter = 0

    for x in np.array(X).astype('float64'):
        x_eq = ski.exposure.equalize_adapthist(x, clip_limit=0.025) # ~6/255
        X_eq.append(x_eq)

        processed_counter += 1
        print(str(processed_counter) + ' images processed', end='\r')

    print(str(processed_counter) + ' images processed')
    return np.array(X_eq, dtype=object)

#%%
# process the training dataset

# determine subfolders
subfolders = list(filter(lambda entry: entry.is_dir(), os.scandir(full_train_path)))

# setup variables to store data into
X_train = []
y_train = []
processed_counter = 0

# process images and annotations for each subfolder
for dir in subfolders:
    # get files in subfolder
    files = list(filter(lambda entry: entry.is_file(), os.scandir(dir.path)))

    # process each file
    for file in files:
        # split into name and extension
        fname_split = os.path.splitext(file.name)
        
        # for image files
        if fname_split[-1].lower() == '.ppm':
            img = ski.io.imread(file.path)
            label = int(dir.name)
            # w = img.shape[1]
            # h = img.shape[0]
            # print("adding image '" + str(file.name) + "' in '"
            #       + str(dir.name) + "' to 'training' (label "
            #       + str(label) + "), " + "(w, h) = " + str((w, h)))
            X_train.append(img)
            y_train.append(label)
            
            processed_counter += 1
            print(str(processed_counter) + ' images processed', end='\r')

print(str(processed_counter) + ' images processed', end='\n')
  
# print and plot some stats
y_train = np.asarray(y_train)
X_train = np.array(X_train, dtype=object)

print('X_train.shape = ' + str(X_train.shape)
    + ', y_train.shape = ' + str(y_train.shape))

_ = plt.hist(y_train, bins=range(np.min(y_train), np.max(y_train) + 2))
plt.title('Histogram of Labels for Training')
plt.xlabel('Class Label')
plt.ylabel('Number of Labels')
plt.grid(visible=True)
plt.show()

# remove images that have less than 1600 pixels in total
print('Rejecting too small images...')
X_train, y_train = reject_small_images(X_train, y_train)

# select some fraction of the images randomly
print('Selecting samples randomly...')
X_train, y_train = select_samples(X_train, y_train)

# show two samples images
plt.imshow(np.asarray(X_train[10], dtype='float32') / 255)
plt.show()
plt.imshow(np.asarray(X_train[3000], dtype='float32') / 255)
plt.show()

# resizing images
print('Resizing images...')
X_train = resize_images(X_train)

# cropping images
print('Cropping images...')
X_train = centercrop_images(X_train)

# equalizing images
print('Equalizing images...')
X_train = equalize_images(X_train)

# show two samples images
plt.imshow(np.asarray(X_train[10], dtype='float32'))
plt.show()
plt.imshow(np.asarray(X_train[3000], dtype='float32'))
plt.show()

print('X_train.shape = ' + str(X_train.shape)
    + ', y_train.shape = ' + str(y_train.shape))

_ = plt.hist(y_train, bins=range(np.min(y_train), np.max(y_train) + 2))
plt.title('Histogram of Labels for Training')
plt.xlabel('Class Label')
plt.ylabel('Number of Labels')
plt.grid(visible=True)
plt.show()

# save to file
print('saving dataset to training.npz...')
np.savez_compressed(os.path.join(path, 'training'),
                    X=X_train, y=y_train)

#%%
# process the testing dataset
files = list(filter(lambda entry: entry.is_file(), os.scandir(full_test_path)))

# find annotations first
annotations_file = None
for file in files:
    fname = file.name
    fname_split = os.path.splitext(fname)

    if fname_split[0].lower().startswith("gt") and fname_split[1].lower() == '.csv':
        annotations_file = file
        break

print('reading annotations file \'' + str(annotations_file.name) + '\'...')
annotations = pd.read_csv(annotations_file.path, sep=';')

print(str(annotations.head(10)))

X_test = []
y_test = []
processed_counter = 0

for i in annotations.index:
    fname = annotations['Filename'][i]

    img_path = os.path.join(full_test_path, fname)
    if os.path.exists(img_path):
        img = ski.io.imread(img_path)
        label = int(annotations['ClassId'][i])
        # w = img.shape[1]
        # h = img.shape[0]
        # print("adding image '" + str(fname)
        #       + "' to 'testing' (label "
        #       + str(label) + "), " + "(w, h) = " + str((w, h)))
        X_test.append(img)
        y_test.append(label)
    
    processed_counter += 1

    if processed_counter % 10 == 0:
        print(str(processed_counter) + ' images processed', end='\r')

print(str(processed_counter) + ' images processed', end='\n')

# print and plot some stats
y_test = np.asarray(y_test)
X_test = np.array(X_test, dtype=object)
print('X_test.shape = ' + str(X_test.shape) 
      + ', y_test.shape = ' + str(y_test.shape))
    
_ = plt.hist(y_test, bins=range(np.min(y_test), np.max(y_test) + 2))
plt.title('Histogram of Labels for Testing')
plt.xlabel('Class Label')
plt.ylabel('Number of Labels')
plt.grid(visible=True)
plt.show()

# remove images that have less than 1600 pixels in total
print('Rejecting too small images...')
X_test, y_test = reject_small_images(X_test, y_test)

# select some fraction of the images randomly
print('Selecting samples randomly...')
X_test, y_test = select_samples(X_test, y_test)

# resizing images
print('Resizing images...')
X_test = resize_images(X_test)

# cropping images
print('Cropping images...')
X_test = centercrop_images(X_test)

# equalizing images
print('Equalizing images...')
X_test = equalize_images(X_test)

# print and plot some stats
y_test = np.asarray(y_test)
X_test = np.array(X_test, dtype=object)
print('X_test.shape = ' + str(X_test.shape)
      + ', y_test.shape = ' + str(y_test.shape))
    
_ = plt.hist(y_test, bins=range(np.min(y_test), np.max(y_test) + 2))
plt.title('Histogram of Labels for Testing')
plt.xlabel('Class Label')
plt.ylabel('Number of Labels')
plt.grid(visible=True)
plt.show()

# save to file
print('saving dataset to testing.npz...')
np.savez_compressed(os.path.join(path, 'testing'),
                    X=X_test, y=y_test)

# %%
