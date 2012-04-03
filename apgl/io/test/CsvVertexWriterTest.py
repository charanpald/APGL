import unittest
import numpy
import logging
from apgl.graph.DictGraph import DictGraph
from apgl.io.CsvVertexWriter import CsvVertexWriter
from apgl.util.PathDefaults import PathDefaults

class CsvVertexWriterTest(unittest.TestCase):
    def setUp(self):
        pass

    def testWriteToFile(self):
        graph = DictGraph()

        numVertices = 5
        numFeatures = 3

        V = numpy.random.rand(numVertices, numFeatures)

        for i in range(0, numVertices):
            graph.setVertex(i, V[i, :])

        fileName = PathDefaults.getOutputDir() + "test/vertices"
        verterWriter = CsvVertexWriter()
        verterWriter.writeToFile(fileName, graph)

        logging.debug(V)



if __name__ == '__main__':
    unittest.main()

