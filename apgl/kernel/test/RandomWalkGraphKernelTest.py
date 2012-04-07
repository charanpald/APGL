from apgl.graph.VertexList import VertexList
from apgl.graph.DenseGraph import DenseGraph
from apgl.graph.SparseGraph import SparseGraph
from apgl.kernel.RandWalkGraphKernel import RandWalkGraphKernel 

import unittest
import numpy
import logging


#TODO: Test this class more thoroughly 
class RandomWalkGraphKernelTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testEvaluate(self):
        numVertices = 6
        numFeatures = 1
        vList = VertexList(numVertices, numFeatures)

        g1 = DenseGraph(vList)
        g1.addEdge(0, 1)
        g1.addEdge(2, 1)
        g1.addEdge(3, 1)
        g1.addEdge(4, 1)
        g1.addEdge(5, 2)

        g2 = DenseGraph(vList)
        g2.addEdge(0, 2)
        g2.addEdge(1, 2)
        g2.addEdge(3, 2)
        g2.addEdge(4, 2)
        g2.addEdge(5, 1)

        g3 = DenseGraph(vList)
        g3.addEdge(0, 1)

        g4 = SparseGraph(vList)
        g4.addEdge(0, 1)
        g4.addEdge(2, 1)
        g4.addEdge(3, 1)
        g4.addEdge(4, 1)
        g4.addEdge(5, 2)

        lmbda = 0.01
        pgk = RandWalkGraphKernel(lmbda)

        logging.debug((pgk.evaluate(g1, g1)))
        logging.debug((pgk.evaluate(g1, g2)))
        logging.debug((pgk.evaluate(g2, g1)))
        logging.debug((pgk.evaluate(g2, g2)))
        logging.debug((pgk.evaluate(g1, g3)))        
        logging.debug((pgk.evaluate(g3, g3)))

        #Tests - graph kernel is symmetric, permutations of indices are identical 
        self.assertAlmostEquals(pgk.evaluate(g1, g2), pgk.evaluate(g2, g1), places=6)
        self.assertAlmostEquals(pgk.evaluate(g1, g3), pgk.evaluate(g3, g1), places=6)
        self.assertAlmostEquals(pgk.evaluate(g1, g1), pgk.evaluate(g1, g2), places=6)
        self.assertAlmostEquals(pgk.evaluate(g2, g4), pgk.evaluate(g1, g2), places=6)

        #All evaluation of themselves are above zero
        self.assertTrue(pgk.evaluate(g1, g1) >= 0)
        self.assertTrue(pgk.evaluate(g2, g2) >= 0)
        self.assertTrue(pgk.evaluate(g3, g3) >= 0)

    #USe this to test in more detail, with special cases etc.
    #Test 1: work out random work explicity and compare
    #Test 2: compare subgraphs
    #Test 3: some other stuff 
    def testEvaluate2(self):
        pass 




if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()