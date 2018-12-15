import matplotlib.pyplot as plt
import numpy as np
import math
import random

"""
def initEnergy(ra, L):
    for a in range(L):
        for b in range(L):
            if (ra[a][b] != 1):
                print("Zero: " + str(a) + "," + str(b))
                continue
            print("One: " + str(a) + "," + str(b))
            for c in range(a, L):
                for d in range(L):
                    if (c == a and d <= b):
                        print("fail1 " + str(c) + "," + str(d))
                        continue
                    if (ra[c][d] == 0):
                        print("fail2 " + str(c) + "," + str(d))
                        continue
                    print("success"  + str(c) + "," + str(d))

x1 = np.array([[0, 1, 0, 1],
                [1, 0, 1, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1]])
initEnergy(x1, 4)
"""
"""
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
        print("Row: " + str(a))
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

x1 = np.array([[0, 1, 0, 1],
                [1, 0, 1, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1]])
initEnergy(x1, 4)
"""
"""
print(3 // 2)
"""

"""
x = 4
x = 4 // 3
print(x)
"""
random.seed(200)
for i in range(12):
    for j in range(3):
        x = random.randint(0, 30)
        print(x)