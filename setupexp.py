
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("exp.sandbox.predictors.TreeCriterion", ["exp/sandbox/predictors/TreeCriterion.pyx"]),
    Extension("exp.util.SparseUtilsCython", ["exp/util/SparseUtilsCython.pyx"]), 
    Extension("exp.util.LinOperatorExpCython", ["exp/util/LinOperatorExpCython.pyx"])
]

setup(
  name = 'Experimental',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)