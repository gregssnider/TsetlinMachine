"""
To create library.so file:

python setup_orig.py build_ext --inplace

"""


from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


setup(
    ext_modules=[
        Extension("OriginalMultiClassTsetlinMachine", ["OriginalMultiClassTsetlinMachine.c"],
                  include_dirs=[numpy.get_include()]),
    ],
)

# Or, if you use cythonize() to make the ext_modules list,
# include_dirs can be passed to setup()

setup(
    ext_modules=cythonize("OriginalMultiClassTsetlinMachine.pyx"),
    include_dirs=[numpy.get_include()]
)