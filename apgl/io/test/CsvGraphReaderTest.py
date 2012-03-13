import logging
import sys
import unittest

from apgl.io.CsvGraphReader import CsvGraphReader
from apgl.util.PathDefaults import PathDefaults
import numpy


class CsvGraphReaderTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    def testReadFromFile(self):
        vertex1Indices = [0, 2, 3, 4, 5]
        vertex2Indices = [1, 6, 7, 8, 9]

        def genderConv(x):
            genderDict = {'"M"': 0, '"F"': 1}
            return genderDict[x]

        def orientConv(x):
            orientDict = {'"HT"': 0, '"HB"': 1}
            return orientDict[x]

        converters = {2: genderConv, 6: genderConv, 3:orientConv, 7:orientConv}

        csvGraphReader = CsvGraphReader(vertex1Indices, vertex2Indices, converters)

        dir = PathDefaults.getDataDir()
        fileName = dir + "test/infect5.csv"

        graph = csvGraphReader.readFromFile(fileName)

        self.assertTrue((graph.getVertex(0) == numpy.array([0, 0, 28, 1])).all())
        self.assertTrue((graph.getVertex(1) == numpy.array([1, 0, 26, 1])).all())
        self.assertTrue((graph.getVertex(2) == numpy.array([0, 1, 42, 2])).all())
        self.assertTrue((graph.getVertex(3) == numpy.array([1, 0, 33, 1])).all())
        self.assertTrue((graph.getVertex(4) == numpy.array([0, 1, 35, 37])).all())

        self.assertTrue(graph.getEdge(0, 1) == 1)
        self.assertTrue(graph.getEdge(2, 3) == 1)
        self.assertTrue(graph.getEdge(4, 6) == 1)
        self.assertTrue(graph.getEdge(6, 7) == 1)
        self.assertTrue(graph.getEdge(5, 8) == 1)

        self.assertEquals(graph.getNumEdges(), 5)
        self.assertTrue(graph.isUndirected())

        #Test a directed graph
        csvGraphReader = CsvGraphReader(vertex1Indices, vertex2Indices, converters, undirected=False)
        graph = csvGraphReader.readFromFile(fileName)

        self.assertTrue(graph.getEdge(1, 0) == None)
        self.assertTrue(graph.getEdge(3, 2) == None)
        self.assertTrue(graph.getEdge(6, 4) == None)
        self.assertTrue(graph.getEdge(7, 6) == None)
        self.assertTrue(graph.getEdge(8, 5) == None)

        self.assertEquals(graph.getNumEdges(), 5)
        self.assertFalse(graph.isUndirected())

        #Test graph with no vertex information
        vertex1Indices = [0]
        vertex2Indices = [1]
        fileName = dir + "test/infect5-0.csv"
        csvGraphReader = CsvGraphReader(vertex1Indices, vertex2Indices, {})
        graph = csvGraphReader.readFromFile(fileName)

        self.assertTrue(graph.getEdge(0, 1) == 1)
        self.assertTrue(graph.getEdge(2, 3) == 1)
        self.assertTrue(graph.getEdge(4, 6) == 1)
        self.assertTrue(graph.getEdge(6, 7) == 1)
        self.assertTrue(graph.getEdge(5, 8) == 1)

        self.assertEquals(graph.getNumEdges(), 5)
        self.assertTrue(graph.isUndirected())
        self.assertEquals(graph.getVertexList().getNumFeatures(), 0)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()