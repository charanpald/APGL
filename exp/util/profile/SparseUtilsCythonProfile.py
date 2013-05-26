

import numpy
import logging
import sys
import scipy.sparse.linalg
import scipy.io
from apgl.util.ProfileUtils import ProfileUtils
from exp.util.SparseUtilsCython import SparseUtilsCython
from exp.util.SparseUtils import SparseUtils
from apgl.util.PathDefaults import PathDefaults 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class SparseUtilsCythonProfile(object):
    def __init__(self):
        numpy.random.seed(21)        
        
    def profilePartialReconstructValsPQ(self):
        shape = 5000, 10000
        r = 100 
        U, s, V = SparseUtils.generateLowRank(shape, r)
        
        k = 1000000 
        inds = numpy.unravel_index(numpy.random.randint(0, shape[0]*shape[1], k), dims=shape)
        
        ProfileUtils.profile('SparseUtilsCython.partialReconstructValsPQ(inds[0], inds[1], U, V)', globals(), locals())

        
profiler = SparseUtilsCythonProfile()
profiler.profilePartialReconstructValsPQ()
