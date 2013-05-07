import numpy
import logging
import sys
import scipy.sparse
from apgl.util.Sampling import Sampling 
from apgl.util.ProfileUtils import ProfileUtils
from exp.sandbox.recommendation.IterativeSoftImpute import IterativeSoftImpute
from exp.util.SparseUtils import SparseUtils

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class IterativeSoftImputeProfile(object):
    def __init__(self):
        numpy.random.seed(21)        
        
        #Create a low rank matrix  
        n = 2000 
        m = 2000 
        self.r = 50 
        k = 10**3
        
        self.X = SparseUtils.generateSparseLowRank((n, m), self.r, k)
        print(self.X.nnz)
        
    def profileModelSelect(self):
        lmbdas = numpy.linspace(1.0, 0.01, 5)
        softImpute = IterativeSoftImpute(k=500)
        
        folds = 5
        cvInds = Sampling.randCrossValidation(folds, self.X.nnz)
        ProfileUtils.profile('softImpute.modelSelect(self.X, lmbdas, cvInds)', globals(), locals())

profiler = IterativeSoftImputeProfile()
profiler.profileModelSelect()
