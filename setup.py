#!/usr/bin/env python
#The first line is needed for egg files and the second for tars etc.
from setuptools import setup
#from distutils.core import setup

#Support for python 3
def execfile3(file, globals=globals(), locals=locals()):
    with open(file, "r") as fh:
        exec(fh.read()+"\n", globals, locals)

execfile3('apgl/version.py')

#Call either of
#python setup.py bdist_egg
#python setup.py sdist

setup (
  name = 'apgl',
  version = __version__,
  #package_dir={'apgl': 'src/apgl'},
  packages = ['apgl', 'apgl.generator', 'apgl.generator.test', 'apgl.predictors', 'apgl.predictors.test', 'apgl.data', 'apgl.features', 'apgl.features.test', 'apgl.graph', 'apgl.graph.test', 'apgl.io', 'apgl.util', 'apgl.util.test', 'apgl.kernel', 'apgl.kernel.test'],

  # Declare your packages' dependencies here, for eg:
  install_requires=['numpy>=1.5.0', 'scipy>=0.7.1'],

  # Fill in these to make your Egg ready for upload to
  # PyPI
  author = 'Charanpal Dhanjal',
  author_email = 'charanpal@gmail.com',
  url = 'http://packages.python.org/apgl/',
  license = 'GNU Library or Lesser General Public License (LGPL)',
  long_description= 'More details are given on the homepage of this project http://packages.python.org/apgl/ .',

  # could also include long_description, download_url, classifiers, etc.
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    ],

  keywords=['graph library', 'numpy', 'scipy', 'machine learning'],
  platforms=["OS Independent"],



)
