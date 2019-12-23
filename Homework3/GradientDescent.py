import numpy as np
import matplotlib.pyplot as plt
import time
import random

start = time.clock()
x = 0.7
y = 0.25
learning_rate = 0.01

w = np.array([x, y])

w_x = []
w_y = []
f = []
threshold = 0.001
while (w[0] + w[1] < 1) and (w[0] > 0) and (w[1] > 0):
    energy = - np.log(1 - w[0] - w[1]) - np.log(w[0]) - np.log(w[1])
    f.append(energy)

    w_x.append(w[0])
    w_y.append(w[1])

    grad_x = 1 / (1 - w[0] - w[1]) - 1 / (w[0])
    grad_y = 1 / (1 - w[0] - w[1]) - 1 / (w[1])
    gradient = np.array([grad_x, grad_y])

    update = learning_rate * gradient

    if np.linalg.norm(w - np.subtract(w, update)) < threshold:
        break
    else:
        w = np.subtract(w, update)

end = time.clock()
print("Time taken for gradient descent: ", round((end - start), 4))

fig, ax = plt.subplots(figsize=(5,5))

plt.scatter(w_x, w_y, c = 'green')
plt.ylim([0,1])
plt.xlim([0,1])
plt.ylabel('Y values')
plt.xlabel('X values')
plt.title('Gradient Descent')

plt.show()

fig, ax = plt.subplots(figsize=(5,5))

plt.scatter(range(len(f)), f , c = 'blue')
plt.ylabel('Energies')
plt.xlabel('Iterations')
plt.title('Energies - Gradient Descent')

plt.show()