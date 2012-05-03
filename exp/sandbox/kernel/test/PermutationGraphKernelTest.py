# To change this template, choose Tools | Templates
# and open the template in the editor.

import unittest
import numpy 
from exp.sandbox.kernel.PermutationGraphKernel import PermutationGraphKernel
from apgl.kernel.LinearKernel import LinearKernel
from apgl.graph.SparseGraph import SparseGraph
from apgl.graph.VertexList import VertexList
from apgl.util.Util import Util

class  PermutationGraphKernelTest(unittest.TestCase):
    def setUp(self):
        self.tol = 10**-4
        self.numVertices = 5
        self.numFeatures = 2

        vertexList1 = VertexList(self.numVertices, self.numFeatures)
        vertexList1.setVertex(0, numpy.array([1, 1]))
        vertexList1.setVertex(1, numpy.array([1, 2]))
        vertexList1.setVertex(2, numpy.array([3, 2]))
        vertexList1.setVertex(3, numpy.array([4, 2]))
        vertexList1.setVertex(4, numpy.array([2, 6]))

        vertexList2 = VertexList(self.numVertices, self.numFeatures)
        vertexList2.setVertex(0, numpy.array([1, 3]))
        vertexList2.setVertex(1, numpy.array([7, 2]))
        vertexList2.setVertex(2, numpy.array([3, 22]))
        vertexList2.setVertex(3, numpy.array([54, 2]))
        vertexList2.setVertex(4, numpy.array([2, 34]))

        self.sGraph1 = SparseGraph(vertexList1)
        self.sGraph1.addEdge(0, 1)
        self.sGraph1.addEdge(0, 2)
        self.sGraph1.addEdge(1, 2)
        self.sGraph1.addEdge(2, 3)

        self.sGraph2 = SparseGraph(vertexList2)
        self.sGraph2.addEdge(0, 1)
        self.sGraph2.addEdge(0, 2)
        self.sGraph2.addEdge(1, 2)
        self.sGraph2.addEdge(2, 3)
        self.sGraph2.addEdge(3, 4)

        self.sGraph3 = SparseGraph(vertexList2)
        self.sGraph3.addEdge(4, 1)
        self.sGraph3.addEdge(4, 2)
        self.sGraph3.addEdge(1, 2)
        self.sGraph3.addEdge(1, 0)


    def testEvaluate(self):
        tau = 1.0
        linearKernel = LinearKernel()

        graphKernel = PermutationGraphKernel(tau, linearKernel)

        """
        First tests - if the graphs have identical edges then permutation is identity matrix
        provided that tau = 1. 
        """

        (evaluation, f, P, SW1, SW2, SK1, SK2) = graphKernel.evaluate(self.sGraph1, self.sGraph1, True)
        self.assertTrue(numpy.linalg.norm(P - numpy.eye(self.numVertices)) <= self.tol)

        S1, U = numpy.linalg.eigh(self.sGraph1.getWeightMatrix())
        S2, U = numpy.linalg.eigh(self.sGraph2.getWeightMatrix())

        evaluation2 = numpy.dot(S1, S1)

        self.assertTrue(numpy.linalg.norm(SW1 - S1) <= self.tol)
        self.assertTrue(numpy.linalg.norm(SW2 - S1) <= self.tol)
        self.assertTrue(abs(evaluation - evaluation2) <= self.tol)

        (evaluation, f, P, SW1, SW2, SK1, SK2) = graphKernel.evaluate(self.sGraph2, self.sGraph2, True)
        self.assertTrue(numpy.linalg.norm(P - numpy.eye(self.numVertices)) <= self.tol)

        evaluation2 = numpy.dot(S2, S2)

        self.assertTrue(numpy.linalg.norm(SW1 - S2) <= self.tol)
        self.assertTrue(numpy.linalg.norm(SW2 - S2) <= self.tol)
        self.assertTrue(abs(evaluation - evaluation2) <= self.tol)

        #Test symmetry
        self.assertEquals(graphKernel.evaluate(self.sGraph1, self.sGraph2), graphKernel.evaluate(self.sGraph2, self.sGraph1))

        #Now we choose tau != 1
        tau = 0.5
        graphKernel = PermutationGraphKernel(tau, linearKernel)

        (evaluation, f, P, SW1, SW2, SK1, SK2) = graphKernel.evaluate(self.sGraph1, self.sGraph1, True)
        self.assertTrue(numpy.linalg.norm(P - numpy.eye(self.numVertices)) <= self.tol)

        self.assertTrue(graphKernel.evaluate(self.sGraph1, self.sGraph1) >= 0)
        self.assertTrue(graphKernel.evaluate(self.sGraph2, self.sGraph2) >= 0) 
        self.assertTrue((graphKernel.evaluate(self.sGraph1, self.sGraph2)- graphKernel.evaluate(self.sGraph2, self.sGraph1)) <= self.tol)

        (evaluation, f, P, SW1, SW2, SK1, SK2) = graphKernel.evaluate(self.sGraph1, self.sGraph2, True)

        self.assertTrue(numpy.linalg.norm(numpy.dot(P.T, P) - numpy.eye(self.numVertices)) <= self.tol)

        #Choose tau=0
        tau = 0.0
        graphKernel = PermutationGraphKernel(tau, linearKernel)

        (evaluation, f, P, SW1, SW2, SK1, SK2) = graphKernel.evaluate(self.sGraph1, self.sGraph1, True)
        self.assertTrue(numpy.linalg.norm(P - numpy.eye(self.numVertices)) <= self.tol)
        self.assertTrue(numpy.linalg.norm(numpy.dot(P.T, P) - numpy.eye(self.numVertices)) <= self.tol)

        X1 = self.sGraph1.getVertexList().getVertices(list(range(0, (self.sGraph1.getNumVertices()))))
        X2 = self.sGraph2.getVertexList().getVertices(list(range(0, (self.sGraph2.getNumVertices()))))
        S1, U = numpy.linalg.eigh(numpy.dot(X1, X1.T))
        S2, V = numpy.linalg.eigh(numpy.dot(X2, X2.T))

        evaluation2 = numpy.dot(S1, S1)

        self.assertTrue(numpy.linalg.norm(SK1 - S1) <= self.tol)
        self.assertTrue(numpy.linalg.norm(SK2 - S1) <= self.tol)
        self.assertTrue(abs(evaluation - evaluation2) <= self.tol)

        self.assertTrue((graphKernel.evaluate(self.sGraph1, self.sGraph2)- graphKernel.evaluate(self.sGraph2, self.sGraph1)) <= self.tol)

    #Test value is zero when we have a graph which is a permutation of the next
    def testEvaluate2(self):
        tau = 1.0
        linearKernel = LinearKernel()

        graphKernel = PermutationGraphKernel(tau, linearKernel)

        (evaluation, f, P, SW1, SW2, SK1, SK2) = graphKernel.evaluate(self.sGraph1, self.sGraph3, True)

        W1 = self.sGraph1.getWeightMatrix()
        W2 = self.sGraph3.getWeightMatrix()

        self.assertTrue(numpy.linalg.norm(Util.mdot(P, W1, P.T)-W2) <= self.tol)
        self.assertAlmostEquals(f, 0, 7)

if __name__ == '__main__':
    unittest.main()

