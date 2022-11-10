import os
from os.path import join
import sys
import glob
import logging
import platform
from setuptools import setup, find_packages, Extension


from Cython.Build import cythonize
from Cython.Distutils import build_ext


NAME = 'musmtl'
VERSION = '0.0.1'


use_openmp = True


def define_extensions():
    if sys.platform.startswith('win'):
        # compile args from
        # https://msdn.microsoft.com/en-us/library/fwkeyyhe.aspx
        compile_args = ['/O2', '/openmp']
        link_args = []
    else:
        gcc = extract_gcc_binaries()
        if gcc is not None:
            rpath = "/usr/local/opt/gcc/lib/gcc/" + gcc[-1] + "/"
            link_args = ["-W1,-rpath," + rpath]
        else:
            link_args = []

    compile_args = ["-Wno-unused-function", "-Wno-maybe-uninitialized", "-O3", "-ffast-math"]
    if use_openmp:
        compile_args.append("-fopenmp")
        link_args.append("-fopenmp")

    compile_args.append("-std=c++11")
    link_args.append("-std=c++11")

    # src_ext = ".pyx"
    modules = [
        Extension(
            f"musmtl.preprocess.{cython_module}._{cython_module}",
            [join("musmtl", "preprocess", f"{cython_module}", f"_{cython_module}.pyx")],
            language="c++",
            extra_compile_args=compile_args,
            extra_link_args=link_args
        )
        for cython_module in ['plsa']
    ]

    # return cythonize(modules)
    return modules


# set_gcc copied from glove-python project
# https://github.com/maciejkula/glove-python

def extract_gcc_binaries():
    """Try to find GCC on OSX for OpenMP support."""
    patterns = [
        "/opt/local/bin/g++-mp-[0-9]*.[0-9]*",
        "/opt/local/bin/g++-mp-[0-9]*",
        "/usr/local/bin/g++-[0-9]*.[0-9]*",
        "/usr/local/bin/g++-[0-9]*",
    ]
    if platform.system() == "Darwin":
        gcc_binaries = []
        for pattern in patterns:
            gcc_binaries += glob.glob(pattern)
        gcc_binaries.sort()
        if gcc_binaries:
            _, gcc = os.path.split(gcc_binaries[-1])
            return gcc
        else:
            return None
    else:
        return None


def set_gcc():
    """Try to use GCC on OSX for OpenMP support."""
    # For macports and homebrew
    if platform.system() == "Darwin":
        gcc = extract_gcc_binaries()

        if gcc is not None:
            os.environ["CC"] = gcc
            os.environ["CXX"] = gcc

        else:
            global use_openmp
            use_openmp = False
            logging.warning(
                "No GCC available. Install gcc from Homebrew " "using brew install gcc."
            )


set_gcc()

def readme():
    with open('README.md') as f:
        return f.read()


def requirements():
    with open('requirements.txt') as f:
        extra_deps = []
        requires = []
        for line in f:
            if line.startswith('--'):
                extra_deps.append(
                    line
                    .replace('\n', '')
                    .replace('--extra-index-url ', '')
                )
            else:
                requires.append(line.strip())
    return requires, extra_deps


setup(name=NAME,
      version=VERSION,
      description='Codebase and utilities for using models trained by multiple music related tasks.',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Information Analysis'
      ],
      keywords=['Multitask Learning', 'Machine Learning', 'Transfer Learning'],
      url='http://github.com/eldrin/MTLMusicRepresentation-PyTorch',
      author='Jaehun Kim',
      author_email='j.h.kim@tudelft.nl',
      license='MIT',
      include_package_data=True,
      packages=find_packages(),
      install_requires=requirements()[0],
      dependency_links=requirements()[1],
      setup_requires=["setuptools>=18.0", "Cython>=0.24"],
      extras_require={
          'dev': [],
      },
      ext_modules=define_extensions(),
      cmdclass={"build_ext": build_ext},
      entry_points = {
          'console_scripts': [
              'mtlextract=musmtl.tool:main',
              'mtltrain=musmtl.experiment:main',
              'mtlutils=musmtl.utils:main'
          ],
      },
      test_suite='tests',
      zip_safe=False)
