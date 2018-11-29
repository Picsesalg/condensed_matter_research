import matplotlib.pyplot as plt
import numpy as np
import random
import datetime
import math

def initLattice(ra, L, numParts):
    now = datetime.datetime.now()
    seed = now.hour * 3600 + now.minute * 60 + now.second
    random.seed(seed)
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

def chooseRandCoord(ra, i, j, L):
    now = datetime.datetime.now()
    seed = now.hour * 3600 + now.minute * 60 + now.second
    random.seed(seed)
    if (i + charge < L and ra[i + charge][j] == 0):
        s = i + charge
        t = j
    elif (i - charge >= 0 and ra[i - charge][j] == 0):
        s = i - charge
        t = j
    elif (j + charge < L and ra[i][j + charge] == 0):
        s = i
        t = j + charge
    elif (j - charge >= 0 and ra[i][j - charge] == 0):
        s = i
        t = j - charge
    else:
        while (ra[i, j] != 1):
            i = random.randint(0, L - 1)
            j = random.randint(0, L - 1)
        s = i
        t = j
    return s, t

def initEnergy(ra, L):
    pivot = L // 2
    offsetA = 0
    offsetB = 0
    offsetC = 0
    offsetD = 0
    term1 = 0
    term2 = 0
    distance = 0
    for a in range(L):
        for b in range(L):
            if (ra[a][b] != 1):
                continue
            offsetA = pivot - a
            offsetB = b - pivot
            term2 = term2 + offsetA ** 2 + offsetB ** 2
            for c in range(a, L):
                for d in range(L):
                    if (c == a and d <= b):
                        continue
                    if (ra[c][d] == 0):
                        continue
                    offsetC = pivot - c
                    offsetD = d - pivot
                    distance = (offsetA - offsetC) ** 2 + (offsetB - offsetD) ** 2
                    distance = distance ** 0.5
                    term1 = term1 - math.log(distance)
    return term1, term2

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
# Number of features on one side of the lattice structure.
L = 31
# Number of inputs.
ni = L * L
# Number of particles
numParts = ni / 2
itval = int(numParts / 2)
charge = 3

# Initalise all elements to 0
x1 = np.zeros((L, L))

countParts = 0
engy_init = 0
engy_init_1 = 0
engy_init_2 = 0
engy_fina = 0

# Initialise the lattice
x1 = initLattice(x1, L, numParts)

# Monte Carlo sampling of 100 steps to converge into the phases.
for m in range (2001):
    for k in range (itval):
# Calculate initial energy
        engy_init_1, engy_init_2 = initEnergy(x1, L)
        engy_init_1 = engy_init_1 * (charge ** 2)
        engy_init_2 = (charge / 4.0) * engy_init_2
        engy_init = engy_init_1 + engy_init_2
# Choose random particle to move
        i = random.randint(0, L - 1)
        j = random.randint(0, L - 1)
        while (x1[i][j] != 1):
            i = random.randint(0, L - 1)
            j = random.randint(0, L - 1)
        s, t = chooseRandCoord(x1, i, j, L)
        x1[i][j] = 0
        x1[s][t] = 1
# Calculate new energy



row = np.array([])
col = np.array([])

for y in range(L):
    for x in range(L):
        if (x1[x][y] == 1):
            count += 1
            row = np.append(row, x)
            col = np.append(col, y)

plt.scatter(row, col)
plt.show()