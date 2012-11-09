

import logging
import unittest
import numpy
import sys
from apgl.graph.DictGraph import DictGraph
from apgl.util.Util import Util
from apgl.util.PathDefaults import PathDefaults
from exp.clusterexp.CitationIterGenerator import CitationIterGenerator

class  CitationIterGeneratorTest(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(suppress=False, precision=3, linewidth=150)
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    #@unittest.skip("demonstrating skipping")
    def testGetIterator(self):
        generator = CitationIterGenerator()
        iterator = generator.getIterator()

        lastW = iterator.next()

        for W in iterator:
            self.assertTrue((W-W.T).getnnz() == 0)
            self.assertTrue((lastW - W[0:lastW.shape[0], 0:lastW.shape[0]]).getnnz() ==0  )
            lastW = W

        numVertices = W.shape[0]

        #Now compute the vertexIds manually:
        dataDir = PathDefaults.getDataDir() + "cluster/"
        edgesFilename = dataDir + "Cit-HepTh.txt"
        dateFilename = dataDir + "Cit-HepTh-dates.txt"

        #We can't load in numbers using numpy since some may start with zero 
        edges = []
        file = open(edgesFilename, 'r')
        file.readline()
        file.readline()
        file.readline()
        file.readline()

        for line in file:
            (vertex1, sep, vertex2) = line.partition("\t")
            vertex1 = vertex1.strip()
            vertex2 = vertex2.strip()
            edges.append([int("1" + vertex1), int("1" + vertex2)])

        edges = numpy.array(edges, numpy.int)

        #Check file read correctly
        self.assertTrue((edges[0, :] == numpy.array([11001, 19304045])).all())
        self.assertTrue((edges[1, :] == numpy.array([11001, 19308122])).all())
        self.assertTrue((edges[9, :] == numpy.array([11001, 19503124])).all())
        vertexIds1 = numpy.unique(edges)
        logging.info("Number of graph vertices: " + str(vertexIds1.shape[0]))

        file = open(dateFilename, 'r')
        file.readline()
        vertexIds2 = []

        for line in file:
            (id, sep, date) = line.partition("\t")
            id = id.strip()
            date = date.strip()
            vertexIds2.append(int("1" + id))

        #Check file read correctly 
        vertexIds2 = numpy.array(vertexIds2, numpy.int)
        self.assertTrue((vertexIds2[0:10] == numpy.array([19203201, 19203202, 19203203, 19203204, 19203205, 19203206, 19203207, 19203208, 19203209, 19203210], numpy.int)).all())
        vertexIds2 = numpy.unique(numpy.array(vertexIds2, numpy.int))

        graph = DictGraph(False)
        graph.addEdges(edges)

        #Find the set of vertices with known citation
        vertices = []
        vertexId2Set = set(vertexIds2.tolist())
        for i in graph.getAllVertexIds():
            Util.printIteration(i, 50000, edges.shape[0])
            if i in vertexId2Set:
                vertices.append(i)
                vertices.extend(graph.neighbours(i))

        logging.debug("Number of final vertices: " + str(numVertices))
        numVertices2 = numpy.unique(numpy.array(vertices)).shape[0]
        self.assertEquals(numVertices, numVertices2)

        #Now compare the weight matrices using the undirected graph
        #Note the order of vertices is different from the iterator 
        graph = DictGraph()
        graph.addEdges(edges)
        subgraph = graph.subgraph(numpy.unique(numpy.array(vertices)))
        W2 = subgraph.getSparseWeightMatrix()

        self.assertEquals(W.getnnz(), W2.getnnz())

    def testEdgeFile(self):
        """
        Figure out the problem with the edge file 
        """
        dataDir = PathDefaults.getDataDir() + "cluster/"
        edgesFilename = dataDir + "Cit-HepTh.txt"

        edges = {}
        file = open(edgesFilename, 'r')
        file.readline()
        file.readline()
        file.readline()
        file.readline()

        vertices = {}

        for line in file:
            (vertex1, sep, vertex2) = line.partition("\t")
            vertex1 = vertex1.strip()
            vertex2 = vertex2.strip()
            edges[(vertex1, vertex2)] = 0
            vertices[vertex1] = 0
            vertices[vertex2] = 0

        #It says there are 352807 edges in paper and 27770 vertices
        self.assertEquals(len(edges), 352807)
        self.assertEquals(len(vertices), 27770)
        
if __name__ == '__main__':
    unittest.main()
