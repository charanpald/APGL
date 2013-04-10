import numpy
import logging
import sys
from apgl.graph import *
from apgl.generator import *
from apgl.util.ProfileUtils import ProfileUtils
from exp.sandbox.recommendation.SoftImpute import SoftImpute
import scipy.sparse
from exp.util.SparseUtils import SparseUtils

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class SoftImputeProfile(object):
    def __init__(self):
        numpy.random.seed(21)        
        
        #Create a low rank matrix  
        n = 100000 
        m = 100000 
        self.r = 50 
        k = 5*10**6
        #k = 10**5
        
        self.X = SparseUtils.generateSparseLowRank((n, m), self.r, k)
        print(self.X.nnz)
        
    def profileLearnModel(self):
        lmbdas = numpy.array([0.5])
        softImpute = SoftImpute(lmbdas)
        
        ProfileUtils.profile('softImpute.learnModel(self.X, False)', globals(), locals())

profiler = SoftImputeProfile()
profiler.profileLearnModel() # 227s 
