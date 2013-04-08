import numpy
import logging
import sys
from apgl.graph import *
from apgl.generator import *
from apgl.util.ProfileUtils import ProfileUtils
from exp.sandbox.recommendation.SoftImpute import SoftImpute
import scipy.sparse

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class SoftImputeProfile(object):
    def __init__(self):
        numpy.random.seed(21)        
        
        #Create a low rank matrix  
        n = 2000 
        m = 2000 
        density = 0.05 
        self.r = 50 
        
        A = numpy.random.rand(n, n)
        U, R = numpy.linalg.qr(A)
        
        B = numpy.random.rand(m, m)
        V, R = numpy.linalg.qr(B)
        
        U = U[:, 0:self.r]
        V = V[:, 0:self.r]
        s = numpy.random.rand(self.r)
        
        X = (U*s).dot(V.T)
        
        numVals = n*m*density 
        print(X.shape)
        
        rowInds = numpy.random.randint(0, n, numVals)
        colInds = numpy.random.randint(0, m, numVals)
        XTrain = numpy.zeros(X.shape)
        XTrain[(rowInds, colInds)] = X[(rowInds, colInds)]
        
        self.XTrain = scipy.sparse.lil_matrix(XTrain)
        self.X = scipy.sparse.lil_matrix(X)
        
        print(self.XTrain.nnz)
        
    def profileLearnModel(self):
        lmbdas = numpy.array([0.1])
        softImpute = SoftImpute(lmbdas)
        
        ProfileUtils.profile('softImpute.learnModel(self.XTrain)', globals(), locals())

profiler = SoftImputeProfile()
profiler.profileLearnModel() # 1.3s 
