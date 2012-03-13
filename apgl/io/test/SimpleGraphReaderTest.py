

import unittest
import logging
from apgl.io.SimpleGraphReader import SimpleGraphReader
from apgl.util import * 


class  SimpleGraphReaderTest(unittest.TestCase):
    def testReadGraph(self):
        fileName = PathDefaults.getDataDir() +  "test/simpleGraph.txt"

        graphReader = SimpleGraphReader()
        graph = graphReader.readFromFile(fileName)

        logging.debug((graph.getAllEdges()))

        self.assertEquals(graph.isUndirected(), True)
        self.assertEquals(graph.getNumVertices(), 5)
        self.assertEquals(graph.getNumEdges(), 4)

        self.assertEquals(graph.getEdge(0, 1), 1)
        self.assertEquals(graph.getEdge(2, 4), 1)
        self.assertEquals(graph.getEdge(2, 2), 1)
        self.assertEquals(graph.getEdge(4, 0), 1)

        #Now test reading a file with the same graph but vertices indexed differently
        fileName = PathDefaults.getDataDir() + "test/simpleGraph2.txt"
        graph = graphReader.readFromFile(fileName)

        self.assertEquals(graph.isUndirected(), True)
        self.assertEquals(graph.getNumVertices(), 5)
        self.assertEquals(graph.getNumEdges(), 4)

        self.assertEquals(graph.getEdge(0, 1), 1.1)
        self.assertEquals(graph.getEdge(2, 4), 1)
        self.assertEquals(graph.getEdge(2, 2), 1.6)
        self.assertEquals(graph.getEdge(4, 0), 1)

        #Now test a file with directed edges
        fileName = PathDefaults.getDataDir() +  "test/simpleGraph3.txt"
        graph = graphReader.readFromFile(fileName)

        self.assertEquals(graph.isUndirected(), False)
        self.assertEquals(graph.getNumVertices(), 5)
        self.assertEquals(graph.getNumEdges(), 4)

        self.assertEquals(graph.getEdge(0, 1), 1)
        self.assertEquals(graph.getEdge(2, 4), 1)
        self.assertEquals(graph.getEdge(2, 2), 1)
        self.assertEquals(graph.getEdge(4, 0), 1)


if __name__ == '__main__':
    unittest.main()

