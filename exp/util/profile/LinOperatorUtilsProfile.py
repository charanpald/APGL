
import numpy
import logging
import sys
import scipy.sparse.linalg
from apgl.util.Sampling import Sampling 
from apgl.util.ProfileUtils import ProfileUtils
from exp.util.SparseUtils import SparseUtils
from exp.util.LinOperatorUtils import LinOperatorUtils

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class LinOperatorUtilsProfile(object):
    def __init__(self):
        numpy.random.seed(21)        
        
        #Create a low rank matrix  
        n = 100000 
        m = 100000 
        self.r = 200 
        k = 10**6
        
        self.X = SparseUtils.generateSparseLowRank((n, m), self.r, k)

    def profileParallelSparseOp(self):
        L = LinOperatorUtils.parallelSparseOp(self.X)
        
        def run(): 
            numRuns = 10 
            for i in range(numRuns): 
                p = numpy.random.rand(self.X.shape[0])
                q = numpy.random.rand(self.X.shape[1])
                
                L.matvec(q)
                L.rmatvec(p)
        
        ProfileUtils.profile('run()', globals(), locals())

    def profileParallelSparseOp2(self):
        L = LinOperatorUtils.parallelSparseOp(self.X)
        
        def run(): 
            numRuns = 10 
            for i in range(numRuns): 
                p = 200
                W = numpy.random.rand(self.X.shape[1], p)
                
                L.matmat(W)
        
        ProfileUtils.profile('run()', globals(), locals())
        
        
    def profileAsLinearOperator(self):
        L = scipy.sparse.linalg.aslinearoperator(self.X)
        
        def run(): 
            numRuns = 10 
            for i in range(numRuns): 
                p = numpy.random.rand(self.X.shape[0])
                q = numpy.random.rand(self.X.shape[1])
                
                L.matvec(q)
                L.rmatvec(p)
        
        ProfileUtils.profile('run()', globals(), locals())

    def profileAsLinearOperator2(self):
        L = scipy.sparse.linalg.aslinearoperator(self.X)
        
        def run(): 
            numRuns = 10 
            for i in range(numRuns): 
                p = 200
                W = numpy.random.rand(self.X.shape[1], p)
                
                L.matmat(W)
        
        ProfileUtils.profile('run()', globals(), locals())
        
profiler = LinOperatorUtilsProfile()
profiler.profileParallelSparseOp()
#profiler.profileAsLinearOperator()
#profiler.profileParallelSparseOp2() # 78.77
#profiler.profileAsLinearOperator2() #44 