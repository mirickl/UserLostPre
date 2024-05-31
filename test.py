import numpy as np

d = np.loadtxt(r"D:\联通\数仓建设\tesxt.txt", usecols=(0, 1, 2, 3, 4, 5), dtype=float)

# (bn,cn) = d.shape

print(type(d))
