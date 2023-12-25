from setuptools import Extension, setup
import numpy as np

ext = Extension("fastroc", ["fastrocmodule.c"], include_dirs=[np.get_include()])
setup(name="fastroc", ext_modules=[ext])