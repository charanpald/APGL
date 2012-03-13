# To change this template, choose Tools | Templates
# and open the template in the editor.

import unittest
import logging
from apgl.graph.VertexList import VertexList
from apgl.graph.SparseGraph import SparseGraph
from apgl.sandbox.GeometricRandomGenerator import GeometricRandomGenerator

class GeometricRandomGeneratorTest(unittest.TestCase):
    def testGenerateGraph(self):
        numFeatures = 0
        numVertices = 20

        vList = VertexList(numVertices, numFeatures)
        graph = SparseGraph(vList)

        alpha1 = 10.0
        alpha2 = 20.0
        p = 0.001
        dim = 2
        generator = GeometricRandomGenerator(graph)

        graph = generator.generateGraph(alpha1, p, dim)
        numEdges1 = graph.getNumEdges()

        #Check no self edges
        for i in range(numVertices):
            self.assertTrue(graph.getEdge(i, i) == None)

        graph.removeAllEdges()
        graph = generator.generateGraph(alpha2, p, dim)
        numEdges2 = graph.getNumEdges()

        #self.assertTrue(numEdges1 >= numEdges2)
        logging.debug(numEdges1)
        logging.debug(numEdges2)

        for i in range(numVertices):
            self.assertTrue(graph.getEdge(i, i) == None)

        #Test case with p=0 and alpha huge 
        p = 0.0
        alpha = 100.0
        graph.removeAllEdges()
        graph = generator.generateGraph(alpha, p, dim)

        self.assertEquals(graph.getNumEdges(),  0)

        #When alpha=0, should get max edges
        alpha = 0.0
        graph.removeAllEdges()
        graph = generator.generateGraph(alpha, p, dim)

        #self.assertEquals(graph.getNumEdges(), int(0.5*(numVertices + numVertices**2) - numVertices))

        #TODO: Test variations in dimension 

        """
        try:
            import networkx
            import matplotlib
        except ImportError as error:
            logging.debug(error)
            pass

        #Show the graph
        D = generator.getPositions()

        nodePositions = {}
        for i in range(numVertices):
            nodePositions[i] = D[i, 0:2]

        nxGraph = graph.toNetworkXGraph()
        nodePositions = networkx.spring_layout(nxGraph)
        nodesAndEdges = networkx.draw_networkx(nxGraph, pos=nodePositions)
        matplotlib.pyplot.show()
        """

    def testDegreeDistribution(self):
        numFeatures = 0
        numVertices = 100

        vList = VertexList(numVertices, numFeatures)
        graph = SparseGraph(vList)

        alpha = 10.0
        p = 0.01
        dim = 2
        generator = GeometricRandomGenerator(graph)
        graph = generator.generateGraph(alpha, p, dim)

        logging.debug((graph.degreeDistribution()))

if __name__ == '__main__':
    unittest.main()

