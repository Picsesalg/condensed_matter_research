import matplotlib.pyplot as plt
import numpy as np

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
print(3 // 2)
"""

x = 4
x = 4 // 3
print(x)