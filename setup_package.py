from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name = 'grid_pairs app',
    ext_modules = cythonize("grid_pairs.pyx"),
    include_dirs = [np.get_include()], 
)

#to compile cython code type: python setup_package.py build_ext --inplace