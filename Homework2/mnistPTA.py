import os
import path
import struct
import numpy as np
import matplotlib.pyplot as plt


# Source for reading the idx files as numpy arrays: https://gist.github.com/tylerneylon
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


# Data Source: http://yann.lecun.com/exdb/mnist/
train_data = read_idx('train-images.idx3-ubyte')
train_labels = read_idx('train-labels.idx1-ubyte')
test_data = read_idx('t10k-images.idx3-ubyte')
test_labels = read_idx('t10k-labels.idx1-ubyte')

def step_fn(x):
    i = 0
    y = np.empty([10,1])
    for each in x:
        if each >= 0:
            y[i] = 1.0
        else:
            y[i] = 0.0
        i += 1
    return y

w = np.random.uniform(-1, 1, size=(10,784))
n = 50
epoch = 0
threshold = 0.0
learning_rate = 1.0
errors = []
