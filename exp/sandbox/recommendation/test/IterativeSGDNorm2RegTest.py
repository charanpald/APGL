
from apgl.util.Util import Util
from apgl.util.Sampling import Sampling 
from exp.util.MCEvaluator import MCEvaluator
from exp.sandbox.recommendation.IterativeSGDNorm2Reg import IterativeSGDNorm2Reg
import sys
import numpy
import unittest
import logging
import scipy.sparse 
import numpy.linalg 
import numpy.testing as nptst 
import exp.util.SparseUtils as ExpSU

class IterativeSGDNorm2RegTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        numpy.set_printoptions(precision=4, suppress=True, linewidth=200)
        
        numpy.seterr(all="raise")
        numpy.random.seed(21)
        
        #Create a sequence of matrices 
        n = 20 
        m = 10 
        r = 8
        U, s, V = ExpSU.SparseUtils.generateLowRank((n, m), r)
        
        numInds = 200
        inds = numpy.random.randint(0, n*m-1, numInds)
        inds = numpy.unique(inds)
        numpy.random.shuffle(inds)
        numInds = inds.shape[0]
        
        inds1 = inds[0:numInds/3]
        inds2 = inds[0:2*numInds/3]
        inds3 = inds
        
        X1 = ExpSU.SparseUtils.reconstructLowRank(U, s, V, inds1)
        X2 = ExpSU.SparseUtils.reconstructLowRank(U, s, V, inds2)
        X3 = ExpSU.SparseUtils.reconstructLowRank(U, s, V, inds3)
        
        self.matrixList = [X1, X2, X3]
        self.indsList = [inds1, inds2, inds3]
        
    def testLearnModel(self): 
        k = 5
        lmbda = 0.01
        eps = 0.000001         
        tmax = 1000        
        
        learner = IterativeSGDNorm2Reg(k, lmbda, eps, tmax)
        
        results = learner.learnModel(iter(self.matrixList)) 
        
        #for Z in results: 
        #    print(Z)
    
    def testPredict(self): 
        k = 5
        lmbda = 0.01
        eps = 0.000001         
        tmax = 1000        
        
        learner = IterativeSGDNorm2Reg(k, lmbda, eps, tmax)
        
        ZList = learner.learnModel(iter(self.matrixList)) 
        
        indList = []
        for X in self.matrixList: 
            indList.append(X.nonzero())

        XList = learner.predict(ZList, indList)    

        for i, Xhat in enumerate(XList): 
            #print(Xhat)
            print(MCEvaluator.rootMeanSqError(Xhat, self.matrixList[i]))
            
    def testModelSelect(self):
        ks = [3,4,5]
        lmbdas = [0.001, 0.01, 0.1, 1]
        gammas = [0.1, 1, 10]
        eps = 0.000001         
        tmax = 1000
        nFolds = 3
        maxNTry = 2
        
        learner = IterativeSGDNorm2Reg(ks[0], lmbdas[0], eps, tmax)
        
        learner.modelSelect(self.matrixList[0], ks, lmbdas, gammas, nFolds, maxNTry) 
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    