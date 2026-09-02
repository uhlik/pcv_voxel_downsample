import sys
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

if sys.platform == "win32":
    extra_compile_args = ["/O2", "/std:c++14"]
else:
    extra_compile_args = ["-O3", "-std=c++11"]

extensions = [
    Extension(
        name="pcv_voxel_downsample",
        sources=["_voxel_downsample.pyx"],
        include_dirs=[np.get_include(), "."],
        language="c++",
        extra_compile_args=extra_compile_args,
    )
]

setup(
    name="pcv_voxel_downsample",
    version="1.1.0",
    ext_modules=cythonize(extensions),
)
