from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='preprocess_cpp',
    ext_modules=[
        cpp_extension.CppExtension(
            'preprocess_cpp', ['preprocess.cpp'])
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)

"""python setup.py build_ext --inplace"""
