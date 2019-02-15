import matplotlib.pyplot as plt
import numpy as np
import random
import datetime
import math

# Initialising random variables.
now = datetime.datetime.now()
seed = now.hour * 3600 + now.minute * 60 + now.second
random.seed(seed)
np.random.seed(seed)

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
    #random.seed(seed)
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

def calcEnergy(ra, L, a, b):
    pivot = L // 2
    offsetA = pivot - a
    offsetB = b - pivot
    offsetC = 0
    offsetD = 0
    term1 = 0
    term2 = offsetA ** 2 + offsetB ** 2
    distance = 0
    for c in range(L):
        for d in range(L):
            if (ra[c][d] == 0):
                continue
            if (c == a and d == b):
                continue
            offsetC = pivot - c
            offsetD = d - pivot
            distance = (offsetA - offsetC) ** 2 + (offsetB - offsetD) ** 2
            distance = distance ** 0.5
            term1 = term1 - math.log(distance)
    return term1, term2

"""
Main function
"""

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
engy_fina_1 = 0
engy_fina_2 = 0
engy_fina = 0

# Initialise the lattice
x1 = initLattice(x1, L, numParts)

row = np.array([])
col = np.array([])

for y in range(L):
    for x in range(L):
        if (x1[x][y] == 1):
            row = np.append(row, x)
            col = np.append(col, y)

plt.scatter(row, col)
plt.show()

# Monte Carlo sampling of 100 steps to converge into the phases.
for m in range (2000):
    for k in range (1):
# Choose random particle to move
        i = random.randint(0, 30)
        j = random.randint(0, 30)
        while (x1[i][j] != 1):
            i = random.randint(0, 30)
            j = random.randint(0, 30)
        engy_init_1, engy_init_2 = calcEnergy(x1, L, i, j)
        engy_init_1 = engy_init_1 * (charge ** 2)
        engy_init_2 = (charge / 4.0) * engy_init_2
        engy_init = engy_init_1 + engy_init_2
        s, t = chooseRandCoord(x1, i, j, L)
        x1[i][j] = 0
        x1[s][t] = 1
# Calculate new energy
        engy_fina_1, engy_fina_2 = calcEnergy(x1, L, s, t)
        engy_fina_1 = engy_fina_1 * (charge ** 2)
        engy_fina_2 = (charge / 4.0) * engy_fina_2
        engy_fina = engy_fina_1 + engy_fina_2
# Decide whether to accept new config
# Keep the flip if new engy is lowered
        x1[i][j] = 1
        x1[s][t] = 0
        if (engy_fina < engy_init):
            x1[i][j] = 0
            x1[s][t] = 1
        else:
# Move with probability by Boltzmann weight 
            boltzmann = -(2 / charge) * (engy_fina - engy_init)
            boltzmann = math.exp(boltzmann)
            if (random.random() < boltzmann):
                x1[i][j] = 1
                x1[s][t] = 0


row = np.array([])
col = np.array([])

for y in range(L):
    for x in range(L):
        if (x1[x][y] == 1):
            row = np.append(row, x)
            col = np.append(col, y)

plt.scatter(row, col)
plt.show()