#!/usr/bin/env python

import numpy as np
from grid_pairs import npairs
from time import time

Lbox = 250.0

Npts = 2e5
x = np.random.uniform(0, Lbox, Npts)
y = np.random.uniform(0, Lbox, Npts)
z = np.random.uniform(0, Lbox, Npts)
rbins = np.logspace(-1, 1, 10)
period = np.array([250.0,250.0,250.0])
data1 = np.vstack((x,y,z))

start = time()
result = npairs(data1,data1,rbins,period=period)
print result
end = time()
runtime = end-start
print("Total runtime = %.1f seconds" % runtime)