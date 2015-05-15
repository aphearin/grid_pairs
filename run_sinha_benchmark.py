#!/usr/bin/env python

import numpy as np
from grid_pairs import npairs
from time import time
import pstats, cProfile

Lbox = 400.0
Npts = 2e5
data1 = np.random.uniform(0, Lbox, Npts*3).reshape(3, Npts)

rmax = 24
rbins = np.logspace(-1, np.log10(rmax), num=10)

start = time()
result = npairs(data1,data1,rbins,Lbox)
print result
end = time()
runtime = end-start
print("\nTotal runtime = %.1f seconds\n" % runtime)

