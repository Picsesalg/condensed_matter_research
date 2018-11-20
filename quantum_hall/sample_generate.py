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

"""
Initialising variables
"""
#Number of features on one side of the lattice structure.
L = 31
#Number of inputs.
ni = L * L
#Number of particles
nump = ni / 2
itval = int(nump / 2)
charge = 3

#Initalise all elements to 0
x1 = np.zeros((L, L))

cp = 0
engy_fina_1 = 0
engy_fina_2 = 0

while (cp < nump):
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
        for a in range(L):
            for b in range(L):
                if x1[a][b] == 1:
                    for c in range(a, L):
                        for d in range(L):
                            if ((c == a and d > b) or c > a):
                                if (x1[c][d] == 1 and a != c and b != d):
                                    distance = (c - a)**2 + (d - b)**2
                                    distance = math.sqrt(distance)
                                    engy_init_1 = engy_init_1 - math.log(distance)
                    engy_init_2 = engy_init_2 + a**2 + b**2
        engy_init_1 = engy_init_1 / 2
        engy_init_1 = charge**2 * engy_init_1
        engy_init_2 = (charge / 4) * engy_init_2
        engy_init = engy_init_1 + engy_init_2

        choice = 0
        s = i
        t = j
        while True:
            choice = choice + 1
            s = i
            t = j
            if (choice == 1):
                s = i + charge
                if (s >= L):
                    s = s - L
            elif (choice == 2):
                s = i - charge
                if (s < 0):
                    s = s + L
            elif (choice == 3):
                t = j + charge
                if (t >= L):
                    t = t - L
            elif (choice == 4):
                t = t - charge
                if (t < 0):
                    t = t + L
            else:
                s = random.randint(0, L - 1)
                t = random.randint(0, L - 1)
            if (x1[s][t] != 1):
                break
        #Calculating new energy configuration
        x1[i][j] = 0
        x1[s][t] = 1
        for a in range(L):  
            for b in range(L):
                if (x1[a][b] == 1):
                    for c in range(a, L):
                        for d in range(L):
                            if ((c == a and d > b) or c > a):
                                if (x1[c][d] == 1 and a != c and b != d):
                                    distance = (c - a)**2 + (d - b)**2
                                    distance = math.sqrt(distance)
                                    engy_fina_1 = engy_fina_1 - math.log(distance)
                    engy_fina_2 = engy_fina_2 + a**2 + b**2
        engy_fina_1 = charge**2 * engy_fina_1
        engy_fina_2 = (charge / 4) * engy_fina_2
        engy_fina = engy_fina_1 + engy_fina_2
        #Accept flip if energy is lowered. Else, keep OG
        if (engy_fina > engy_init):
            x1[i][j] = 1
            x1[s][t] = 0
        elif (random.random() > np.exp(-2 * engy_fina / charge)):
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