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
    y = np.empty([10, 1])
    for each in x:
        if each >= 0:
            y[i] = 1.0
        else:
            y[i] = 0.0
        i += 1
    return y


def update_weight(w):
    for i in range(n):
        xi = train_data[i]
        xi.resize(784, 1)
        y = np.array(step_fn(np.matmul(w, xi)))
        label = np.zeros((1, 10)).T
        label[train_labels[i]] = 1
        difference = label - y
        xit = np.transpose(xi)
        update = learning_rate * np.matmul(difference, xit)
        w += update


def training_errors(epoch, errors):
    for i in range(n):
        xi = train_data[i]
        xi.resize(784, 1)
        v = np.matmul(w, xi)
        prediction = v.argmax(axis=0)
        actual = train_labels[i]
        if prediction != actual:
            errors[epoch] += 1
    return errors[epoch]


def learning_weights(w, epoch, threshold, learning_rate):
    while epoch < 100:
        errors.append(0)
        errors[epoch] = training_errors(epoch, errors)
        update_weight(w)
        epoch += 1
        if errors[epoch - 1] / n <= threshold:
            break


w = np.random.uniform(-1, 1, size=(10, 784))
n = 50  # n=1000 # n = 60000
epoch = 0
threshold = 0.0  # Change threshold end with lesser epochs
learning_rate = 1.0
errors = []
learning_weights(w, epoch, threshold, learning_rate)

fig, ax = plt.subplots(figsize=(10, 10))
plt.plot(range(len(errors)), errors, c='green')
plt.ylabel('Number of Misclassifications')
plt.xlabel('Number of Epochs')
plt.title('Errors vs Epochs')
plt.show()

test_errors = 0
for i in range(len(test_data)):
    xi = test_data[i]
    xi.resize(784, 1)
    v = np.matmul(w, xi)
    prediction = v.argmax(axis=0)
    actual = test_labels[i]
    if prediction != actual:
        test_errors += 1
print("Number of errors in test data: ", test_errors)
print("Percentage of test errors: ", test_errors * 100 / len(test_data))
