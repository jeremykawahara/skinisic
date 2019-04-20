import os
from distutils.core import setup
from Cython.Build import cythonize

dir_path = os.path.dirname(os.path.realpath(__file__))

setup(
    name='cython_converter',
    ext_modules=cythonize(os.path.join(dir_path, "cython_converter.pyx")),
)
