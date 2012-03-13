'''
Created on 3 Aug 2009

@author: charanpal
'''
import unittest
import numpy 
from apgl.data.Standardiser import Standardiser
from apgl.data.ExamplesList import ExamplesList


class PreprocessorTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass


    def testNormaliseArray(self):
        numExamples = 10 
        numFeatures = 3 
        
        preprocessor = Standardiser()
        
        #Test an everyday matrix 
        X = numpy.random.rand(numExamples, numFeatures)
        Xn = preprocessor.normaliseArray(X)
        normV = preprocessor.getNormVector()
        self.assertAlmostEquals(numpy.sum(Xn*Xn), numFeatures, places=3)
        
        norms = numpy.sum(Xn*Xn, 0)
        
        for i in range(0, norms.shape[0]): 
            self.assertAlmostEquals(norms[i], 1, places=3)
            
        self.assertTrue((X/normV == Xn).all())
        
        #Zero one column 
        preprocessor = Standardiser()
        X[:, 1] = 0 
        Xn = preprocessor.normaliseArray(X)
        normV = preprocessor.getNormVector()
        self.assertAlmostEquals(numpy.sum(Xn*Xn), numFeatures-1, places=3)
        self.assertTrue((X/normV == Xn).all())
        
        #Now take out 3 rows of X, normalise and compare to normalised X 
        Xs = X[0:3, :]
        Xsn = preprocessor.normaliseArray(Xs)
        self.assertTrue((Xsn == Xn[0:3, :]).all())
        
    def testCentreArray(self):
        numExamples = 10 
        numFeatures = 3 
        
        preprocessor = Standardiser()
        
        #Test an everyday matrix 
        X = numpy.random.rand(numExamples, numFeatures)
        Xc = preprocessor.centreArray(X)
        centreV = preprocessor.getCentreVector()
        self.assertAlmostEquals(numpy.sum(Xc), 0, places=3)
        self.assertTrue((X-centreV == Xc).all())
        
        #Now take out 3 rows of X, normalise and compare to normalised X 
        Xs = X[0:3, :]
        Xsc = preprocessor.centreArray(Xs)
        self.assertTrue((Xsc == Xc[0:3, :]).all())
        
    def testStandardiseArray(self):
        numExamples = 10 
        numFeatures = 3 
        
        preprocessor = Standardiser()
        
        #Test an everyday matrix 
        X = numpy.random.rand(numExamples, numFeatures)
        Xs = preprocessor.standardiseArray(X)
        
        self.assertAlmostEquals(numpy.sum(Xs), 0, places=3)
        self.assertAlmostEquals(numpy.sum(Xs*Xs), numFeatures, places=3)
        
        #Now, test on a portion of a matrix 
        Xss = preprocessor.standardiseArray(X[1:5, :])
        self.assertTrue((Xss == Xs[1:5, :]).all())
        
    def testUnstandardiseArray(self):
        numExamples = 10
        numFeatures = 3

        tol = 10**-6
        preprocessor = Standardiser()

        #Test an everyday matrix
        X = numpy.random.rand(numExamples, numFeatures)
        Xs = preprocessor.standardiseArray(X)
        X2 = preprocessor.unstandardiseArray(Xs)

        self.assertTrue(numpy.linalg.norm(X2 - X) < tol)
        

    def testScaleArray(self):
        numExamples = 10
        numFeatures = 3
        X = numpy.random.rand(numExamples, numFeatures)

        preprocessor = Standardiser()
        Xs = preprocessor.scaleArray(X)

        minVals = numpy.amin(Xs, 0)
        maxVals = numpy.amax(Xs, 0)

        tol = 10**-6
        self.assertTrue(numpy.linalg.norm(minVals + numpy.ones(X.shape[1])) <= tol)
        self.assertTrue(numpy.linalg.norm(maxVals - numpy.ones(X.shape[1])) <= tol)

        #Now test stanrdisation on other matrix

        X = numpy.array([[2, 1], [-1, -2], [0.6, 0.3]])
        preprocessor = Standardiser()
        Xs = preprocessor.scaleArray(X)

        X2 = numpy.array([[2, 1], [-1, -2], [0.6, 0.3], [4, 2]])
        Xs2 = preprocessor.scaleArray(X2)

        self.assertTrue(numpy.linalg.norm(Xs2[0:3, :] - Xs) < tol)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()