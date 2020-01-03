import numpy as np
import matplotlib.pyplot as plt

# Curve fitting using input data and formula
n = 300
x = np.random.uniform(0.0, 1.0, n)
v = np.random.uniform(-0.1, 0.1, n)

d = []
for i in range(n):
    d.append(np.sin(20 * x[i]) + (3 * x[i]) + v[i])

fig, ax = plt.subplots(figsize=(10, 10))
plt.xlabel('x')
plt.ylabel('d')
plt.scatter(x, d, c='blue')
plt.show()


# Using backprop

def act_fun(f): return np.tanh(f)


def act_output(v): return v


def activation_derv(v): return (1 - np.tanh(v) ** 2)


def act_output_derv(v): return 1


def feed_forward(x, w_input, bias, w_output, w_final, N, i):
    v = []
    temp = []

    for j in range(N):
        alpha = (x * w_input[j]) + bias[j]
        temp.append(alpha)
        v.append(act_fun(alpha))
    alphas.append(temp)
    u.append(v)
    beta = np.matmul(np.array(u[i]), w_output) + w_final
    betas.append(beta[0])
    y.append(act_output(beta[0]))
    return y, u, alphas


def back_prop(d, y, eta, N, u, x, w_output, alphas, i):
    e = -((d[i] - y[i]) * eta * 2) / n
    w_input_grad = []
    bias_grad = []
    w_output_grad = []
    w_final_grad = []
    w_final_grad.append(-e)
    for j in range(N):
        w_output_grad.append(e * u[i][j])
        w_input_grad.append(e * x[i] * w_output[j] * activation_derv(alphas[i][j]))
        bias_grad.append(e * w_output[j] * act_output_derv(alphas[i][j]))

    return w_input_grad, bias_grad, w_output_grad, w_final_grad


N = 24
eta = 6
w_input = np.random.uniform(-5, 5, N)
bias = np.random.uniform(-1, 1, N)
w_output = np.random.uniform(-5, 5, N)
w_final = np.random.uniform(-1, 1, 1)

mse_list = []
z = 0
while True:
    y = []
    u = []
    alphas = []
    betas = []
    for i in range(n):
        y, u, alphas = feed_forward(x[i], w_input, bias, w_output, w_final, N, i)
        w_input_grad = []
        bias_grad = []
        w_output_grad = []
        w_final_grad = []
        w_input_grad, bias_grad, w_output_grad, w_final_grad = back_prop(d, y, eta, N, u, x, w_output, alphas, i)
        w_input = np.subtract(w_input, np.asarray(w_input_grad))
        w_output = np.subtract(w_output, np.asarray(w_output_grad))
        bias = np.subtract(bias, np.asarray(bias_grad))
        w_final = np.subtract(w_final, np.asarray(w_final_grad))

    mse = 0
    for i in range(n):
        mse += (d[i] - y[i]) ** 2
    mse = mse / n
    mse_list.append(mse)
    print(mse, eta, z)

    if mse_list[z] > mse_list[z - 1]:
        eta = 0.9 * eta
    if mse_list[-1] < 0.01:
        break
    z += 1

fig, ax = plt.subplots(figsize=(10, 10))
plt.ylabel('Mean Square Error')
plt.xlabel('Epochs')
plt.scatter(range(len(mse_list)), mse_list, c='blue')
plt.show()

u = []
y = []
alphas = []
betas = []
for j in range(n):
    v = []
    temp = []
    for i in range(N):
        alpha = x[j] * w_input[i] + bias[i]
        temp.append(alpha)
        v.append(act_fun(alpha))
    alphas.append(temp)
    u.append(v)
    beta = np.matmul(np.array(u[j]), w_output) + w_final
    betas.append(beta)
    y.append(act_output(beta[0]))

fig, ax = plt.subplots(figsize=(10, 10))
plt.ylabel('d')
plt.xlabel('x')
plt.scatter(x, d, c='red', label='Actual')
plt.scatter(x, y, c='blue', label='Predicted')
plt.legend(loc='best')
plt.show()
