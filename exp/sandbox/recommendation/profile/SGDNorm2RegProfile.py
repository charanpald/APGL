import numpy
import logging
import sys
from apgl.graph import *
from apgl.generator import *
from apgl.util.ProfileUtils import ProfileUtils
#from exp.sandbox.recommendation.SGDNorm2Reg import SGDNorm2Reg
from exp.sandbox.recommendation.SGDNorm2RegCython import SGDNorm2Reg
import scipy.sparse
from exp.util.SparseUtils import SparseUtils

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class SGDNorm2RegProfile(object):
    def __init__(self):
        numpy.random.seed(21)        
        
        #Create a low rank matrix  
        n = 100000 
        m = 100000 
        self.r = 50 
        nKnown = 10**4
        
        self.X = SparseUtils.generateSparseLowRank((n, m), self.r, nKnown)
        print(self.X.nnz)
        
    def profileLearnModel(self, useProfiler=True, eps=10**(-6)):
        lmbdas = numpy.array([0.5])
        k = 100
        lmbda = 0.001
        tmax=100000
        gamma = 1
        
        learner = SGDNorm2Reg(k, lmbda, eps, tmax)
        
        if useProfiler:
            ProfileUtils.profile('learner.learnModel(self.X, storeAll=False)', globals(), locals())
        else:
            learner.learnModel(self.X, storeAll=False)

if __name__ == "__main__":
    profiler = SGDNorm2RegProfile()
    profiler.profileLearnModel()
#    profiler.profileLearnModel(eps=-1.)
    # with n=m=10^6 and nKnown=10^4
    # 26s with python (b461f6296304480e033044c5e41aa04052895206)
    # 25s with python and do not reallocte (340b059cf2fc43944c10ba0773a07a5577f946e0)
    # 12s with python and a nonzero array to replace X[u,i] (340b059cf2fc43944c10ba0773a07a5577f946e0)
    # 11s with Cython
    # 9s with Cython and store deltaPNorm 
    # 6s with nothing in the loop
    # 2.7s with nothing in the loop and no Norm computation
    # Overhead to compute deltaNorm is equivalent to gradient => let's have an option to remove epsilon stopping
    
    # current version
    # 9s with Cython
    # 6.5s with Cython and eps < 0
    # 10.5s with Python
    # 7s with Python and eps < 0
    
