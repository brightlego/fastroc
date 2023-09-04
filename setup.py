from distutils.core import setup, Extension
import numpy as np

ext = Extension("fastroc", ["fastrocmodule.c"], include_dirs=[np.get_include()])

setup(name="fastroc", version="0.0", ext_modules=[ext])