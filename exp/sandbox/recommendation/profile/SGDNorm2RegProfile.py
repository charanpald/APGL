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
#        n = 100000 
#        m = 100000 
#        self.r = 50 
#        nKnown = 10**4
        # netflix-like
#        n = 480000
#        m = 18000 
#        self.r = 200 
#        nKnown = 10**8
        # close from netflix
        n = 480000
        m = 18000 
        self.r = 200 
        nKnown = 10**6
        # focusing on scalar-product
        n = 480000
        m = 18000 
        self.r = 50 
        nKnown = 10**5
        
        self.X = SparseUtils.generateSparseLowRank((n, m), self.r, nKnown)
        print(self.X.nnz)
        
    def profileLearnModel(self, useProfiler=True, eps=10**(-6)):
        k = 100
        lmbda = 0.001
        tmax=10**7
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
    # with n = 480000, m = 18000, self.r = 200, nKnown = 10**6, tmax=10**7, k=100
    # !! norm2 is negligible (4s) !!
    # 45s with Cython
    # 10s with Cython and dot-product by hand.
    
    
    
    
