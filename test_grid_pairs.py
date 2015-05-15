#!/usr/bin/env python

import numpy as np
from grid_pairs import npairs
from time import time
import pstats, cProfile

Lbox = 400.0
rmax = 24

Npts = 2e5
x = np.random.uniform(0, Lbox, Npts)
y = np.random.uniform(0, Lbox, Npts)
z = np.random.uniform(0, Lbox, Npts)
rbins = np.logspace(-1, np.log10(rmax), num=10)
period = np.array([Lbox,Lbox,Lbox])
data1 = np.vstack((x,y,z))

start = time()
result = npairs(data1,data1,rbins,period=period)
print result
end = time()
runtime = end-start
print("Total runtime = %.1f seconds" % runtime)

#cProfile.runctx("npairs(data1,data1,rbins,period=period)", globals(), locals(), "Profile.prof")
#s = pstats.Stats("Profile.prof")
#s.strip_dirs().sort_stats("time").print_stats()