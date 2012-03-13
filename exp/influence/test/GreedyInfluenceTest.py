import unittest
import numpy 
from apgl.influence.GreedyInfluence import GreedyInfluence
from apgl.graph.VertexList import VertexList
from apgl.graph.SparseGraph import SparseGraph 

class  GreedyInfluenceTest(unittest.TestCase):
    def setUp(self):
        pass 

    def testGreedyInfluence(self): 
        numVertices = 4
        P = numpy.zeros((numVertices, numVertices))

        P[0, :] = numpy.array([1, 0.3, 0.4, 0.1])
        P[1, :] = numpy.array([0, 1, 0.5, 0.2])
        P[2, :] = numpy.array([0, 0.1, 1, 0])
        P[3, :] = numpy.array([0.3, 0.6, 0.4, 1])

        k = 2
        influence = GreedyInfluence()
        inds = influence.maxInfluence(P, k)

        self.assertTrue(inds == [3, 0])

        k = 4
        influence = GreedyInfluence()
        inds = influence.maxInfluence(P, k)

        self.assertTrue(inds == [3, 0, 2, 1])

        #Now test case in which we can get best influence in 1 vertex and we
        #want to choose a larger set.
        P[3, :] = numpy.array([1, 1, 1, 1])
        P[1, :] = numpy.array([0, 1, 0.5, 0.2])
        P[2, :] = numpy.array([0, 0.1, 1, 0])
        P[0, :] = numpy.array([0.3, 0.6, 0.4, 1])

        k = 3
        inds = influence.maxInfluence(P, k)

        self.assertTrue(inds == [3, 0, 1])

    def testGraphInfuence(self):
        #We test the influence using a real graph 
        numVertices = 5
        numFeatures = 0

        vList = VertexList(numVertices, numFeatures)
        sGraph = SparseGraph(vList, False)

        sGraph.addEdge(0, 1, 0.1)
        sGraph.addEdge(0, 2, 0.5)
        sGraph.addEdge(1, 3, 0.9)
        sGraph.addEdge(2, 3, 0.7)
        sGraph.addEdge(2, 4, 0.8)

        P = sGraph.maxProductPaths()

        self.assertTrue((P[0, :] == numpy.array([0,0.1, 0.5, 0.35, 0.4])).all())
        self.assertTrue((P[1, :] == numpy.array([0,0,0,0.9,0])).all())
        self.assertTrue((P[2, :] == numpy.array([0,0,0,0.7,0.8])).all())
        self.assertTrue((P[3, :] == numpy.array([0,0,0,0,0])).all())
        self.assertTrue((P[4, :] == numpy.array([0,0,0,0,0])).all())

        k = 5
        influence = GreedyInfluence()
        inds = influence.maxInfluence(P, k)

    def testMaxBudgetedInfluence(self):
        #First test the case where the cost of each vertex is 1 and L = k
        numVertices = 4
        P = numpy.zeros((numVertices, numVertices))
        u = numpy.ones(numVertices)

        P[0, :] = numpy.array([1, 0.3, 0.4, 0.1])
        P[1, :] = numpy.array([0, 1, 0.5, 0.2])
        P[2, :] = numpy.array([0, 0.1, 1, 0])
        P[3, :] = numpy.array([0.3, 0.6, 0.4, 1])

        k = 2.0
        influence = GreedyInfluence()
        inds = influence.maxBudgetedInfluence(P, u, k)

        self.assertTrue(inds == [3, 0])

        k = 4.0
        influence = GreedyInfluence()
        inds = influence.maxBudgetedInfluence(P, u, k)

        self.assertTrue(inds == [3, 0, 2, 1])

        #Now test case in which we can get best influence in 1 vertex and we
        #want to choose a larger set.
        P[3, :] = numpy.array([1, 1, 1, 1])
        P[1, :] = numpy.array([0, 1, 0.5, 0.2])
        P[2, :] = numpy.array([0, 0.1, 1, 0])
        P[0, :] = numpy.array([0.3, 0.6, 0.4, 1])

        k = 3.0
        inds = influence.maxBudgetedInfluence(P, u, k)

        #This result differs from the unbudgeted version since we don't ask for
        #k indices 
        self.assertTrue(inds == [3])

        #Now test the budgeted case with varying cost vectors 
        numVertices = 5
        P = numpy.zeros((numVertices, numVertices))
        u = numpy.array([3,4,6,2,1])
        P[0, :] = numpy.array([2, 8, 3, 4, 1])
        P[1, :] = numpy.array([1, 9, 6, 2, 3])
        P[2, :] = numpy.array([12, 8 ,9, 3, 1])
        P[3, :] = numpy.array([0, 2, 1, 6, 9])
        P[4, :] = numpy.array([1, 0, 0, 8, 2])

        L = 7.0


        inds = influence.maxBudgetedInfluence(P, u, L)

        self.assertEquals(inds, [4, 3, 1])

        L = 8.0
        inds = influence.maxBudgetedInfluence(P, u, L)
        self.assertEquals(inds, [4, 3, 1])

        L = 9.0
        inds = influence.maxBudgetedInfluence(P, u, L)
        self.assertEquals(inds, [4, 3, 2])

        #This is the maximum budget 
        L = 16.0
        inds = influence.maxBudgetedInfluence(P, u, L)
        inds2 = influence.maxInfluence((P.T/u).T, 4)
        self.assertEquals(inds, [4, 3, 2, 0])
        self.assertEquals(inds, inds2)

        P[1, :] = numpy.array([2, 8, 6.5, 4, 1])
        inds = influence.maxBudgetedInfluence(P, u, L)
        self.assertEquals(inds, [4, 3, 2, 0, 1])

if __name__ == '__main__':
    unittest.main()

