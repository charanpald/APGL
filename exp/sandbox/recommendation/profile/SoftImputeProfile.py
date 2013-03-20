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
        n = 400 
        m = 500 
        
        A = numpy.random.rand(n, n)
        A = A.T.dot(A)
        l, U = numpy.linalg.eigh(A)
        
        B = numpy.random.rand(m, m)
        B = B.T.dot(B)
        l, V = numpy.linalg.eigh(B)
        
        r = 50 
        U = U[:, 0:r]
        V = V[:, 0:r]
        s = numpy.random.rand(r)
        
        X = (U*s).dot(V.T)
        
        numVals = n*m*0.1 
        print(X.shape)
        print(numpy.random.randint(0, n, numVals))
        
        rowInds = numpy.random.randint(0, n, numVals)
        colInds = numpy.random.randint(0, m, numVals)
        XTrain = numpy.zeros(X.shape)
        XTrain[(rowInds, colInds)] = X[(rowInds, colInds)]
        
        self.XTrain = scipy.sparse.lil_matrix(XTrain)
        self.X = scipy.sparse.lil_matrix(X)
        
    def profileLearnModel(self):
        lmbdas = numpy.array([0.1])
        softImpute = SoftImpute(lmbdas)
        
        ProfileUtils.profile('softImpute.learnModel(self.XTrain)', globals(), locals())

profiler = SoftImputeProfile()
profiler.profileLearnModel()
