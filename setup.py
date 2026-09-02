import os
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
        name="pcv_voxel_downsample._voxel_downsample",
        sources=["src/pcv_voxel_downsample/_voxel_downsample.pyx"],
        include_dirs=[np.get_include(), "src/pcv_voxel_downsample"],
        language="c++",
        extra_compile_args=extra_compile_args,
    )
]

setup(
    ext_modules=cythonize(extensions, language_level="3"),
)
