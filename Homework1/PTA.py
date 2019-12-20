import numpy as np
import matplotlib.pylab as plt

# %matplotlib inline

w0 = np.random.uniform(-0.25, 0.25)
w1 = np.random.uniform(-1, 1)
w2 = np.random.uniform(-1, 1)

x_dim = 2
n = 1000 #change to 1000 for part p
S = np.random.uniform(-1, 1, size=(n, x_dim))
S0 = []
S1 = []
for i in S:
    if (1*w0)+(i[0]*w1)+(i[1]*w2) >= 0:
            S1.append([i[0]] + [i[1]] + [0])
    elif (i[0]*w1)+(i[1]*w2) < 0:
            S0.append([i[0]] + [i[1]] + [1])


X = np.array([-(w0-w2)/w1, -(w0+w2)/w1])
Y = np.array([-1.0, +1.0])

S1_x = []
S1_y = []
S0_x = []
S0_y = []

for i in S0:
    S0_x.append(i[0])
    S0_y.append(i[1])
for i in S1:
    S1_x.append(i[0])
    S1_y.append(i[1])

fig, ax = plt.subplots(figsize=(10,10))
red = plt.scatter(S0_x, S0_y, c ='r', marker ='*', label='Class 1')
blue = plt.scatter(S1_x, S1_y, c='b',marker ='^', label='Class 0')
line = ax.plot(X, Y, c = 'black', label=' Decision Boundry')
plt.title('3(i)')
plt.legend(loc="upper right")
plt.ylim([-1,1])
plt.xlim([-1,1])
plt.show()

def step(x):
    if x < 0: y = 0
    else: y = 1
    return y

def initialize_weights():
    w0_1 = np.random.uniform(-1, 1)
    w1_1 = np.random.uniform(-1, 1)
    w2_1 = np.random.uniform(-1, 1)
    W = []
    W = [w0_1, w1_1, w2_1]
    return W

dataset = S0 + S1
W = initialize_weights()

def misclassified(dataset, W):
    misclass = 0
    for i in dataset:
        z = (W[0]+(i[0]*W[1])+(i[1]*W[2]))
        y = step(z)
        if y != i[2]:
            misclass += 1
    return misclass
a = misclassified(dataset, W)
print('Number of misclassifications: ', a)

def PTA(weight, learning_rate):
    epoch = 0
    new_W = []
    missed = []
    while (misclassified(dataset,weight)!=0):
        missed.append(misclassified(dataset,weight))
        print('Number of missclassifications: ', missed[epoch])
        epoch += 1
        print('Epoch Number: ', epoch)
        for i in range(len(dataset)):
            z = weight[0] + (dataset[i][0]*weight[1]) + (dataset[i][1]*weight[2])
            y = step(z)
            update =[1]+dataset[i][0:2]
            desired_output = dataset[i][2]
            difference = desired_output-y
            if difference != 0:
                weight[0] = weight[0]+update[0]*learning_rate*difference
                weight[1] = weight[1]+update[1]*learning_rate*difference
                weight[2] = weight[2]+update[2]*learning_rate*difference
        print('Updated weights: ', weight)
        new_W.append(weight)
    final_misclassification = misclassified(dataset,weight)
    print('Number of missclassifications: ', final_misclassification)
    return new_W, missed

learning_rate = 1
print('Initial weight: ' , W)
weights=[]
weights, missed = PTA(W, learning_rate)
n_epochs = range(len(weights)+1)
print(n_epochs)
fig, ax = plt.subplots(figsize=(10,10))
ax.plot(n_epochs, missed+[0], c = 'black')
plt.ylabel('Number of Misclassifications')
plt.xlabel('Number of Epochs')
plt.show()

learning_rate = 10
print('Initial weight: ' , W)
weights=[]
weights, missed= PTA(W, learning_rate)
n_epochs = range(len(weights)+1)
fig, ax = plt.subplots(figsize=(10,10))
ax.plot(n_epochs, missed+[0], c = 'black')
plt.ylabel('Number of Misclassifications')
plt.xlabel('Number of Epochs')
plt.show()

learning_rate = 0.1
print('Initial weight: ' , W)
weights=[]
weights, missed= PTA(W, learning_rate)
n_epochs = range(len(weights)+1)
fig, ax = plt.subplots(figsize=(10,10))
ax.plot(n_epochs, missed+[0], c = 'black')
plt.ylabel('Number of Misclassifications')
plt.xlabel('Number of Epochs')
plt.show()
