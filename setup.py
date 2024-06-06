from setuptools import setup
from Cython.Build import cythonize

ext_modules = [
    "_cosine_k_means_common.pyx",
    "_cosine_k_means_lloyd.pyx",
]

setup(
    ext_modules=cythonize(ext_modules)
)