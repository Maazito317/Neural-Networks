import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import time

start = time.clock()

x = 0.7
y = 0.2
learning_rate = 1

w = np.array([x,y])

w_x = []
w_y = []
f = []
threshold = 0.001

