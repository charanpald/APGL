
from exp.sandbox.ClusterBound import ClusterBound
import unittest
import numpy
import numpy.testing as nptst 
import sys 
import logging

class ClusterBoundTest(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(suppress=True, precision=3, linewidth=150)
        numpy.random.seed(21)
        #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        
    def testComputeKClusterBound(self): 
        numExamples = 100 
        numFeatures = 2
        
        #Test some 2 cluster examples 
        for i in range(50): 
            V = numpy.random.rand(numExamples, numFeatures)
            
            numCluster1 = numpy.random.randint(5, numExamples)
            
            V[0:numCluster1, :] += numpy.random.randn()
            U = V - numpy.mean(V)  
            delta = numpy.linalg.norm(U)*0.1
                    
            k = 2 
            obj, bestSigma = ClusterBound.computeKClusterBound(U, delta, k)
            obj2, bestSigma2 = ClusterBound.compute2ClusterBound(U, delta)
            
            self.assertAlmostEquals(obj, obj2)
            
        #Now use more clusters 
        numExamples = 30 
        numFeatures = 5
        
        V = numpy.zeros((numExamples, numFeatures))
        V[0:10 ,:] = numpy.random.randn(10, numFeatures) + numpy.array([1, 2, -1, 5, -4])
        V[10:20 ,:] = numpy.random.randn(10, numFeatures) + numpy.array([1, 1, -1, 5, -4])
        V[20:30 ,:] = numpy.random.randn(10, numFeatures) + numpy.array([-3, 4, -0.1, 0.5, 2])
        
        U = V - numpy.mean(V)  
        delta = numpy.linalg.norm(U)*0.1
        k = 4 

        #In delta=0 case the sigmas are the same as the a_is 
        for k in range(2, 5): 
            obj, bestSigma = ClusterBound.computeKClusterBound(U, 0, k)

            X, a, Y = numpy.linalg.svd(U)
            a = numpy.flipud(numpy.sort(a))
            self.assertAlmostEquals(obj, (a[k-1:]**2).sum(), 4)
            self.assertAlmostEquals(obj, (bestSigma[k-1:]**2).sum(), 4)

        #delta != 0         
        delta = numpy.linalg.norm(U)*0.1
        obj, bestSigma = ClusterBound.computeKClusterBound(U, 0, k)
        self.assertAlmostEquals(obj, (bestSigma[k-1:]**2).sum(), 4)
        
        #Do some random tests 
        for i in range(20): 
            V = numpy.zeros((numExamples, numFeatures))
            V[0:10 ,:] = numpy.random.randn(10, numFeatures) + numpy.random.rand(numFeatures)
            V[10:20 ,:] = numpy.random.randn(10, numFeatures) + numpy.random.rand(numFeatures)
            V[20:30 ,:] = numpy.random.randn(10, numFeatures) + numpy.random.rand(numFeatures)
            
            U = V - numpy.mean(V)  
            delta = numpy.linalg.norm(U)*0.1
                    
            for k in range(2, 5):   
                obj, bestSigma = ClusterBound.computeKClusterBound(U, delta, k)
                self.assertAlmostEquals(obj, (bestSigma[k-1:]**2).sum(), 4)
                self.assertTrue((bestSigma[0:k-1] >= bestSigma[k-1]).all() and (bestSigma[k:] <= bestSigma[k-1]).all())
        
        #Try on a simple toy example for which we know the answer 
        U = numpy.array([[5, 0], [0, 1]])
        delta = 1
        k = 2
        obj, bestSigma = ClusterBound.computeKClusterBound(U, delta, k)
        self.assertEquals(obj, 4)
        nptst.assert_array_equal(bestSigma, numpy.array([5, 2]))
        
        U = numpy.array([[5, 0], [0, 1]])
        delta = 4
        k = 2
        obj, bestSigma = ClusterBound.computeKClusterBound(U, delta, k)
        self.assertAlmostEquals(obj, 9)
        nptst.assert_array_almost_equal(bestSigma, numpy.array([5, 3]))
        
        #Now try on 3-cluster example 
        U = numpy.array([[5, 0, 0], [0, 2, 0], [0, 0, 1]])
        delta = 0
        k = 3
        obj, bestSigma = ClusterBound.computeKClusterBound(U, delta, k)
        self.assertEquals(obj, 1)
        nptst.assert_array_equal(bestSigma, numpy.array([5, 2, 1]))
        
        U = numpy.array([[5, 0, 0], [0, 2, 0], [0, 0, 1]])
        delta = 1
        k = 3
        obj, bestSigma = ClusterBound.computeKClusterBound(U, delta, k)
        self.assertEquals(obj, 4)
        nptst.assert_array_equal(bestSigma, numpy.array([5, 2, 2]))
        
        U = numpy.array([[5, 0, 0], [0, 2, 0], [0, 0, 1]])
        delta = 2
        k = 3
        obj, bestSigma = ClusterBound.computeKClusterBound(U, delta, k)
        
        #To solve this we look at sigmak(sigmak-4) + sigmak(sigmak-2) = -3
        sigmak = 3.0**(1.0/2.0)/2.0 + 3.0/2.0
        self.assertAlmostEquals(obj, sigmak**2)        
        nptst.assert_array_almost_equal(bestSigma, numpy.array([5, sigmak, sigmak]))
            
    @unittest.skip("")
    def testComputeKClusterBound2(self): 
        #Try strange case where bound is less than continuous solution 
        #ValueError: Bound is smaller than real solution: 127.7443918 29.6138874353
        
        U = numpy.load("/home/dhanjalc/Documents/Postdoc/Code/APGL/repo/exp/sandbox/badMatrix.npy")
        delta = numpy.load("/home/dhanjalc/Documents/Postdoc/Code/APGL/repo/exp/sandbox/badDelta.npy")
        k = 3
        
        obj, bestSigma = ClusterBound.computeKClusterBound(U, delta, k)
        
        X, a, Y = numpy.linalg.svd(U)
        a = numpy.flipud(numpy.sort(a))
        
        self.assertTrue(obj > 127.7443918)
        

if __name__ == '__main__':
    unittest.main()