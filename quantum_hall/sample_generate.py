import matplotlib.pyplot as plt
import numpy as np
import random
import datetime
import math

def initLattice(ra, L, numParts):
    cp = 0
    while (cp < numParts):
        i = random.randint(0, L - 1)
        j = random.randint(0, L - 1)
        while (ra[i, j] == 1):
            i = random.randint(0, L - 1)
            j = random.randint(0, L - 1)
        ra[i, j] = 1
        cp += 1
    return ra

def chooseRandCoord(ra, i, j):
    if (x1[i + charge, j] == 0):
        s = i + charge
           t = j
    elif (x1[i - charge, j] == 0):
        s = i - charge
        t = j
    elif (x1[i, j + charge] == 0):
        s = i
        t = j + charge
    elif (x1[i, j - charge] == 0):
        s = i
        t = j - charge
    else:
        while (x1[i, j] != 1):
            i = random.randint(0, L - 1)
            j = random.randint(0, L - 1)
        s = i
        t = j
    return s, t

"""
Initialising random variables.
"""
now = datetime.datetime.now()
seed = now.hour * 3600 + now.minute * 60 + now.second
random.seed(seed)
np.random.seed(seed)

"""
Initialising variables
"""
#Number of features on one side of the lattice structure.
L = 31
#Number of inputs.
ni = L * L
#Number of particles
numParts = ni / 2
itval = int(numParts / 2)
charge = 3

#Initalise all elements to 0
x1 = np.zeros((L, L))

countParts = 0
engy_fina_1 = 0
engy_fina_2 = 0

#Initialise the lattice
x1 = initLattice(x1, L, numParts)

#Monte Carlo sampling of 100 steps to converge into the phases.
for m in range (2001):
    for k in range (itval):
#Calculate initial energy

#Choose random particle to move
        i = random.randint(0, L - 1)
        j = random.randint(0, L - 1)
        while (x1[i, j] != 1):
            i = random.randint(0, L - 1)
            j = random.randint(0, L - 1)
        x1[i, j] = 0
        s, t = chooseRandCoord(x1, i, j)
        x1[s, t] = 1


row = np.array([])
col = np.array([])

for y in range(L):
    for x in range(L):
        if (x1[x][y] == 1):
            row = np.append(row, x)
            col = np.append(col, y)

plt.scatter(row, col)
plt.show()