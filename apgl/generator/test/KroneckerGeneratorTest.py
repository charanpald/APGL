
import numpy 
from apgl.graph.VertexList import VertexList
from apgl.graph.SparseGraph import SparseGraph
from apgl.generator.KroneckerGenerator import KroneckerGenerator
import unittest
import logging

class KroneckerGeneratorTest(unittest.TestCase):
    def setUp(self):
        pass

    def testGenerate(self):
        k = 2
        numVertices = 3
        numFeatures = 0

        vList = VertexList(numVertices, numFeatures)
        initialGraph = SparseGraph(vList)
        initialGraph.addEdge(0, 1)
        initialGraph.addEdge(1, 2)

        for i in range(numVertices):
            initialGraph.addEdge(i, i)

        d = initialGraph.diameter()
        degreeSequence = initialGraph.outDegreeSequence()
        generator = KroneckerGenerator(initialGraph, k)

        graph = generator.generate()
        d2 = graph.diameter()
        degreeSequence2 = graph.outDegreeSequence()

        self.assertTrue((numpy.kron(degreeSequence, degreeSequence) == degreeSequence2).all())
        self.assertTrue(graph.getNumVertices() == numVertices**k)
        self.assertTrue(graph.getNumDirEdges() == initialGraph.getNumDirEdges()**k)
        self.assertEquals(d, d2)

        #Try different k
        k = 3
        generator.setK(k)
        graph = generator.generate()
        d3 = graph.diameter()
        degreeSequence3 = graph.outDegreeSequence()

        self.assertTrue((numpy.kron(degreeSequence, degreeSequence2) == degreeSequence3).all())
        self.assertTrue(graph.getNumVertices() == numVertices**k)
        self.assertTrue(graph.getNumDirEdges() == initialGraph.getNumDirEdges()**k)
        self.assertEquals(d, d3)

        #Test the multinomial degree distribution
        logging.debug(degreeSequence)
        logging.debug(degreeSequence2)
        logging.debug(degreeSequence3)

    def testDegreeDistribution(self):
        #We want to see how the degree distribution changes with kronecker powers

        
        numVertices = 3
        numFeatures = 0

        vList = VertexList(numVertices, numFeatures)
        initialGraph = SparseGraph(vList)
        initialGraph.addEdge(0, 1)
        initialGraph.addEdge(1, 2)

        for i in range(numVertices):
            initialGraph.addEdge(i, i)

        logging.debug((initialGraph.outDegreeSequence()))
        logging.debug((initialGraph.degreeDistribution()))

        k = 2
        generator = KroneckerGenerator(initialGraph, k)
        graph = generator.generate()

        logging.debug((graph.outDegreeSequence()))
        logging.debug((graph.degreeDistribution()))

        k = 3
        generator = KroneckerGenerator(initialGraph, k)
        graph = generator.generate()

        logging.debug((graph.degreeDistribution()))