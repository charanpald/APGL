'''
Created on 6 Jul 2009

@author: charanpal
'''
import sys
from apgl.io.EgoCsvReader import EgoCsvReader
from exp.egograph.EgoGenerator import EgoGenerator
from apgl.util.Util import Util
import numpy
import unittest
import logging

class EgoGeneratorTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        
    def tearDown(self):
        pass

    def testInit(self):
        pass        
    
    def testGenerateIndicatorVertices(self):
        egoGenerator = EgoGenerator() 
        
        numVertices = 500000
        means = numpy.array([1, 10])
        vars = numpy.array([[5, 1], [1, 2]])
        p = 0.1
        
        vList = egoGenerator.generateIndicatorVertices(numVertices, means, vars, p)
        X = numpy.zeros((numVertices, means.shape[0]+1))
        
        for i in range(0, numVertices): 
            X[i, :] = vList.getVertex(i)
        
        (means2, vars2) = Util.computeMeanVar(X)

        self.assertTrue((X.astype(numpy.int32) == X).all())

        self.assertAlmostEquals(numpy.linalg.norm(means2[0:2] - means), 0, places=1)
        self.assertAlmostEquals(numpy.linalg.norm(vars2[0:2][:,0:2] - vars), 0, places=0)
        self.assertAlmostEquals(p, means2[2],places=2)

        #Try non-symmetric variance matrix
        vars = numpy.array([[5, 1], [8, 2]])
        self.assertRaises(ValueError, egoGenerator.generateIndicatorVertices, numVertices, means, vars, p)

    def testGenerateIndicatorVertices2(self):
        egoGenerator = EgoGenerator()

        numVertices = 500000
        means = numpy.array([1, 10])
        vars = numpy.array([[5, 1], [1, 2]])
        mins = numpy.array([-1000, -1000])
        maxs = numpy.array([1000, 1000])

        p = 0.1

        vList = egoGenerator.generateIndicatorVertices2(numVertices, means, vars, p, mins, maxs)
        X = numpy.zeros((numVertices, means.shape[0]+1))

        for i in range(0, numVertices):
            X[i, :] = vList.getVertex(i)

        (means2, vars2) = Util.computeMeanVar(X)

        self.assertTrue((X.astype(numpy.int32) == X).all())

        self.assertAlmostEquals(numpy.linalg.norm(means2[0:2] - means), 0, places=1)
        self.assertAlmostEquals(numpy.linalg.norm(vars2[0:2][:,0:2] - vars), 0, places=0)
        self.assertAlmostEquals(p, means2[2],places=2)

        self.assertTrue((X[:, 0:2].min(0) >= mins).all())
        self.assertTrue((X[:, 0:2].max(0) <= maxs).all())

        #Try non-symmetric variance matrix
        vars = numpy.array([[5, 1], [8, 2]])
        self.assertRaises(ValueError, egoGenerator.generateIndicatorVertices2, numVertices, means, vars, p, mins, maxs)

        #Test min > max
        vars = numpy.array([[5, 1], [1, 2]])
        mins = numpy.array([-2, 6])
        maxs = numpy.array([10, 5])
        self.assertRaises(ValueError, egoGenerator.generateIndicatorVertices2, numVertices, means, vars, p, mins, maxs)

        #Test min == max
        numVertices = 1000
        vars = numpy.array([[5, 1], [1, 2]])
        mins = numpy.array([-2, 5])
        maxs = numpy.array([10, 5])
        vList = egoGenerator.generateIndicatorVertices2(numVertices, means, vars, p, mins, maxs)

        for i in range(0, numVertices):
            self.assertTrue(vList.getVertex(i)[1] == 5)

        #Try a new example with small range of min and max - check the mean and var
        numVertices = 500000
        means = numpy.array([1, 10])
        vars = numpy.array([[2, 0], [0, 2]])
        mins = numpy.array([-3, 6])
        maxs = numpy.array([5, 14])

        p = 0.1

        vList = egoGenerator.generateIndicatorVertices2(numVertices, means, vars, p, mins, maxs)
        X = numpy.zeros((numVertices, means.shape[0]+1))

        for i in range(0, numVertices):
            X[i, :] = vList.getVertex(i)

        (means2, vars2) = Util.computeMeanVar(X)

        self.assertAlmostEquals(numpy.linalg.norm(means2[0:2] - means), 0, places=1)
        self.assertAlmostEquals(numpy.linalg.norm(vars2[0:2][:,0:2] - vars), 0, places=0)
        self.assertAlmostEquals(p, means2[2],places=2)

        logging.debug((X[:, 0:2].min(0)))
        logging.debug((X[:, 0:2].max(0)))

        self.assertTrue((X[:, 0:2].min(0) >= mins).all())
        self.assertTrue((X[:, 0:2].max(0) <= maxs).all())



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    