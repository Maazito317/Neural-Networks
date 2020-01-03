from __future__ import division
import idx2numpy
import numpy as np
import sys
import pickle
import math
import sys
import matplotlib.pyplot as plt
sys.stdout.flush()


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derv(x):
    return sigmoid(x) * (1 - sigmoid(x))


train_input = idx2numpy.convert_from_file('train-images.idx3-ubyte');
train_doutput = idx2numpy.convert_from_file('train-labels.idx1-ubyte');

eta = 50
n = 60000
n_hidden = 250
batch_size = 2000
n_batch = int(n / batch_size)
weights_l2 = np.random.rand(n_hidden, 785)
d_weights_l2 = np.zeros(shape=(n_hidden, 785))

weights_l3 = np.random.rand(10, n_hidden + 1)
d_weights_l3 = np.zeros(shape=(10, n_hidden + 1))

epoch = 0
convergence = False

for epoch in range(0, 100):
    print('\nEpoch: ' + str(epoch))

    epoch_errors = [0] * 10
    epoch_energies = [0] * 10

    for batch in range(0, n_batch):

        errors = [0] * 10
        energies = [0] * 10

        for s in range(0, batch_size):
            sample_num = batch * batch_size + s

            input_layer = train_input[sample_num].flatten().tolist() + [1]

            post_hidden_layer = np.matrix(weights_l2) * np.matrix(input_layer).T
            post_hidden_layer = post_hidden_layer.flatten().tolist()[0]
            post_hidden_function = [sigmoid(x) for x in post_hidden_layer]

            add_bias = post_hidden_function + [1]

            post_output_layer = np.matrix(weights_l3) * np.matrix(add_bias).T
            post_output_layer = post_output_layer.flatten().tolist()[0]
            post_output_function = [sigmoid(x) for x in post_output_layer]

            desired_out = [0] * 10
            desired_out[train_doutput[sample_num]] = 1

            diff = np.subtract(post_output_function, desired_out)
            errors = np.add(errors, diff)
            epoch_errors = np.add(epoch_errors, errors)
            diff = np.fromiter(map(lambda x: x * x, diff), dtype=np.int)
            energies = np.add(energies, diff)

        epoch_energies = np.add(epoch_energies, energies)

        avg_error = np.fromiter(map(lambda x: x / batch_size, errors), dtype=np.int)
        print('\nAverage batch error: ' + str(batch))

        dE_de = 2 * avg_error
        de_dy3 = -1

        for i in range(0, 10):
            dy3_dv3 = np.fromiter(map(sigmoid_derv, post_output_layer), dtype=np.int)

            for j in range(0, n_hidden):
                dv3_dWeight_j = post_hidden_function[j]

                dv3_dy2 = weights_l3[i, j]
                dy2_dv2 = np.fromiter(map(sigmoid_derv, post_hidden_layer), dtype=np.int)

                d_weights_l3[i, j] += dE_de[i] * de_dy3 * dy3_dv3[i] * dv3_dWeight_j

                for k in range(0, 784):
                    dv2_dWeight_k = input_layer[k]
                    d_weights_l2[j, k] += dE_de[i] * de_dy3 * dy3_dv3[i] * dv3_dy2 * dy2_dv2[j] * dv2_dWeight_k

                dv3_dy2 = weights_l3[i, n_hidden]

                dv2_dWeight_bias = 1
                d_weights_l2[j, 784] += dE_de[i] * de_dy3 * dy3_dv3[i] * dv3_dy2 * dy2_dv2[j] * dv2_dWeight_bias

            dv3_dWeight_bias = 1
            d_weights_l3[i, n_hidden] += dE_de[i] * de_dy3 * dy3_dv3[i] * dv3_dWeight_bias

        for i in range(0, n_hidden):
            for j in range(0, 785):
                weights_l2[i, j] += eta * d_weights_l2[i, j]
                d_weights_l2[i, j] = 0

        for i in range(0, 10):
            for j in range(0, n_hidden + 1):
                weights_l3[i, j] += eta * d_weights_l3[i, j]
                d_weights_l3[i, j] = 0

    epoch_energies = np.fromiter(map(lambda x: x / n, epoch_energies), dtype=np.int)

    f = open('digit_wise_energy.csv', 'a')
    for digit in range(0, 9):
        f.write(str(epoch_energies[digit]) + ',')
    f.write(str(epoch_energies[9]) + '\n')
    f.close()

fig, ax = plt.subplots(figsize=(10, 10))
plt.ylabel('Epoch Errors')
plt.xlabel('Epochs training')
plt.scatter(range(0, 100), epoch_errors, c='blue')
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
plt.ylabel('Epoch Energies')
plt.xlabel('Epochs training')
plt.scatter(range(0, 100), epoch_energies, c='blue')
plt.show()

f = open('weights_l2.pckl', 'wb')
pickle.dump(weights_l2, f)
f.close()
f = open('weights_l3.pckl', 'wb')
pickle.dump(weights_l3, f)
f.close()
