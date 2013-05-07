from apgl.util.Util import Util
from apgl.util.Sampling import Sampling 
from apgl.util.MCEvaluator import MCEvaluator
from exp.sandbox.recommendation.IterativeSoftImpute import IterativeSoftImpute
from exp.sandbox.recommendation.SoftImpute import SoftImpute 
import sys
import numpy
import unittest
import logging
import scipy.sparse 
import numpy.linalg 
import numpy.testing as nptst 
import exp.util.SparseUtils as ExpSU

class IterativeSoftImputeTest(unittest.TestCase):
    def setUp(self):
        #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
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
        lmbda = 0.0 
        eps = 0.1 
        k = 10
        
        matrixIterator = iter(self.matrixList)
        iterativeSoftImpute = IterativeSoftImpute(lmbda, k=k, eps=eps, svdAlg="propack")
        ZList = iterativeSoftImpute.learnModel(matrixIterator)
        
        #Check that ZList is the same as XList 
        for i, Z in enumerate(ZList):
            U, s, V = Z
            Xhat = (U*s).dot(V.T)
            
            nptst.assert_array_almost_equal(Xhat, self.matrixList[i].todense())
        
        #Compare solution with that of SoftImpute class 
        lmbdaList = [0.1, 0.2, 0.5, 1.0]
        
        for lmbda in lmbdaList: 
            iterativeSoftImpute = IterativeSoftImpute(lmbda, k=k, eps=eps, svdAlg="propack", updateAlg="zero")
            
            matrixIterator = iter(self.matrixList)
            ZList = iterativeSoftImpute.learnModel(matrixIterator)
            
            lmbdas = numpy.array([lmbda])
            
            softImpute = SoftImpute(lmbdas, k=k, eps=eps)
            Z1 = softImpute.learnModel(self.matrixList[0])
            Z2 = softImpute.learnModel(self.matrixList[1])
            Z3 = softImpute.learnModel(self.matrixList[2])
            
            ZList2 = [Z1, Z2, Z3]
            
            for j, Zhat in enumerate(ZList):
                U, s, V = Zhat 
                Z = (U*s).dot(V.T)
                nptst.assert_array_almost_equal(Z, ZList2[j].todense())
                
                #Also test with true solution Z = S_lambda(X + Z^\bot_\omega)
                Zomega = numpy.zeros(self.matrixList[j].shape)
                
                rowInds, colInds = self.matrixList[j].nonzero()
                for i in range(self.matrixList[j].nonzero()[0].shape[0]): 
                    Zomega[rowInds[i], colInds[i]] = Z[rowInds[i], colInds[i]]
                    
                U, s, V = ExpSU.SparseUtils.svdSoft(numpy.array(self.matrixList[j]-Zomega+Z), lmbda)      
                
                tol = 0.1
                self.assertTrue(numpy.linalg.norm(Z -(U*s).dot(V.T))**2 < tol)
    

    def testLearnModel2(self): 
        #Test the SVD updating solution in the case where we get an exact solution 
        lmbda = 0.0 
        eps = 0.1 
        k = 20
        
        matrixIterator = iter(self.matrixList)
        iterativeSoftImpute = IterativeSoftImpute(lmbda, k=k, eps=eps, svdAlg="RSVD")
        ZList = iterativeSoftImpute.learnModel(matrixIterator)
        
        #Check that ZList is the same as XList 
        for i, Z in enumerate(ZList):
            U, s, V = Z
            Xhat = (U*s).dot(V.T)
            
            nptst.assert_array_almost_equal(Xhat, self.matrixList[i].todense())
        
        #Compare solution with that of SoftImpute class 
        lmbdaList = [0.1, 0.2, 0.5, 1.0]
        
        for lmbda in lmbdaList: 
            iterativeSoftImpute = IterativeSoftImpute(lmbda, k=k, eps=eps, svdAlg="RSVD", updateAlg="zero")
            
            matrixIterator = iter(self.matrixList)
            ZList = iterativeSoftImpute.learnModel(matrixIterator)
            
            lmbdas = numpy.array([lmbda])
            
            softImpute = SoftImpute(lmbdas, k=k, eps=eps)
            Z1 = softImpute.learnModel(self.matrixList[0])
            Z2 = softImpute.learnModel(self.matrixList[1])
            Z3 = softImpute.learnModel(self.matrixList[2])
            
            ZList2 = [Z1, Z2, Z3]
            
            for j, Zhat in enumerate(ZList):
                U, s, V = Zhat 
                Z = (U*s).dot(V.T)
                nptst.assert_array_almost_equal(Z, ZList2[j].todense())
                
                #Also test with true solution Z = S_lambda(X + Z^\bot_\omega)
                Zomega = numpy.zeros(self.matrixList[j].shape)
                
                rowInds, colInds = self.matrixList[j].nonzero()
                for i in range(self.matrixList[j].nonzero()[0].shape[0]): 
                    Zomega[rowInds[i], colInds[i]] = Z[rowInds[i], colInds[i]]
                    
                U, s, V = ExpSU.SparseUtils.svdSoft(numpy.array(self.matrixList[j]-Zomega+Z), lmbda)      
                
                tol = 0.1
                self.assertTrue(numpy.linalg.norm(Z -(U*s).dot(V.T))**2 < tol)
        
        
        
    def testLearnModel3(self): 
        #Test using Randomised SVD 

        
        #Test on an increasing then decreasing set of solutions 
        pass 

    #@unittest.skip("")
    def testPredict(self): 
        #Create a set of indices 
        lmbda = 0.0 
        
        iterativeSoftImpute = IterativeSoftImpute(lmbda, k=10)
        matrixIterator = iter(self.matrixList)
        ZList = iterativeSoftImpute.learnModel(matrixIterator)
        
        XhatList = iterativeSoftImpute.predict(ZList, self.indsList)
        
        #Check we get the exact matrices returned 
        for i, Xhat in enumerate(XhatList): 
            nptst.assert_array_almost_equal(numpy.array(Xhat.todense()), self.matrixList[i].todense())
            
            self.assertEquals(Xhat.nnz, self.indsList[i].shape[0])
            
        #Try moderate lambda 
        lmbda = 0.1 
        iterativeSoftImpute = IterativeSoftImpute(lmbda, k=10)
        matrixIterator = iter(self.matrixList)
        ZList = list(iterativeSoftImpute.learnModel(matrixIterator)) 
        
        XhatList = iterativeSoftImpute.predict(iter(ZList), self.indsList)
        
        for i, Xhat in enumerate(XhatList): 
            for ind in self.indsList[i]:
                U, s, V = ZList[i]
                Z = (U*s).dot(V.T)
                self.assertEquals(Xhat[numpy.unravel_index(ind, Xhat.shape)], Z[numpy.unravel_index(ind, Xhat.shape)])
            
            self.assertEquals(Xhat.nnz, self.indsList[i].shape[0])

    #@unittest.skip("")
    def testModelSelect(self):
        lmbda = 0.1
        shape = (20, 20) 
        r = 20 
        k = 50
        noise = 0.2
        X = ExpSU.SparseUtils.generateSparseLowRank(shape, r, k, noise)
        
        U, s, V = numpy.linalg.svd(X.todense())

        iterativeSoftImpute = IterativeSoftImpute(lmbda, k=None, svdAlg="propack", updateAlg="zero")
        lmbdas = numpy.linspace(numpy.max(s)+0.1, 0.001, 20)
        folds = 5 
        cvInds = Sampling.randCrossValidation(folds, X.nnz)
        meanTestErrors = iterativeSoftImpute.modelSelect(X, lmbdas, cvInds)

        #Now do model selection manually 
        (rowInds, colInds) = X.nonzero()
        trainErrors = numpy.zeros((lmbdas.shape[0], len(cvInds)))
        testErrors = numpy.zeros((lmbdas.shape[0], len(cvInds)))
        
        for i, lmbda in enumerate(lmbdas): 
            for j, (trainInds, testInds) in enumerate(cvInds): 
                trainX = scipy.sparse.csc_matrix(X.shape)
                testX = scipy.sparse.csc_matrix(X.shape)
                
                for p in trainInds: 
                    trainX[rowInds[p], colInds[p]] = X[rowInds[p], colInds[p]]
                    
                for p in testInds: 
                    testX[rowInds[p], colInds[p]] = X[rowInds[p], colInds[p]]
                                 
                softImpute = SoftImpute(numpy.array([lmbda])) 
                ZList = [softImpute.learnModel(trainX, fullMatrices=False)]
                
                predTrainX = softImpute.predict(ZList, trainX.nonzero())[0]
                predX = softImpute.predict(ZList, testX.nonzero())[0]

                
                testErrors[i, j] = MCEvaluator.meanSqError(testX, predX)
                trainErrors[i, j] = MCEvaluator.meanSqError(trainX, predTrainX)
        
        meanTestErrors2 = testErrors.mean(1)   
        meanTrainErrors2 = trainErrors.mean(1)  
        nptst.assert_array_almost_equal(meanTestErrors, meanTestErrors2, 3) 

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    