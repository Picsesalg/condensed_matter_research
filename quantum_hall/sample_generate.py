import matplotlib.pyplot as plt
import numpy as np
import random
import datetime
import math

"""
Initialising random variables.
"""
now = datetime.datetime.now()
seed = now.hour * 3600 + now.minute * 60 + now.second
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

"""
Initialising variables
"""
#Number of features on one side of the lattice structure.
L = 31
#Number of inputs.
ni = L * L
#Number of particles
np = ni / 2
itval = np / 2
charge = 3

#Initalise all elements to 0
x1 = np.zeros((L, L))

cp = 0

while (cp < np):
    i = random.randint(0, L - 1)
    j = random.randint(0, L - 1)
    if (x1[i][j] == 0):
        x1[i][j] = 1
        cp = cp + 1

for m in range(2001):
    for n in range(itval):
        while True:
            i = random.randint(0, L - 1)
            j = random.randint(0, L - 1)
            if (x1[i][j] == 1):
                break
        engy_init_1 = 0
        engy_init_2 = 0
        for 