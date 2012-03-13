import unittest
import os
import numpy 
from apgl.io.MultiGraphCsvReader import MultiGraphCsvReader
from apgl.util.PathDefaults import PathDefaults 

class MultiGraphCsvReaderTest(unittest.TestCase):
    def setUp(self):
        pass

    def testReadGraph(self):

        dir = PathDefaults.getDataDir()
        vertexFileName = dir + "test/deggraf10.csv"
        edgeFileNames = [dir + "test/testEdges1.csv", dir + "test/testEdges2.csv"]

        def genderConv(x):
            genderDict = {'"M"': 0, '"F"': 1}
            return genderDict[x]

        def orientConv(x):
            orientDict = {'"HT"': 0, '"HB"': 1}
            return orientDict[x]

        def fteConv(x):
            fteDict = {'"INTER"': 0, '"CONTA"': 1}
            return fteDict[x]

        def provConv(x):
            provDict = {'"CH"': 0, '"SC"': 1, '"SS"': 2, '"LH"' : 3, '"GM"' : 4}
            return provDict[x]

        converters = {3: genderConv, 4: orientConv, 5:fteConv, 6:provConv}

        idIndex = 0
        featureIndices = list(range(1,11))
        multiGraphCsvReader = MultiGraphCsvReader(idIndex, featureIndices, converters)
        sparseMultiGraph = multiGraphCsvReader.readGraph(vertexFileName, edgeFileNames)

        vertexValues = numpy.zeros((10, 10))
        vertexValues[0, :] = numpy.array([1986, 32, 0, 0, 0, 0, 0, 3, 3, 1])
        vertexValues[1, :] = numpy.array([1986, 27, 0, 0, 0, 1, 0, 4, 4, 1])
        vertexValues[2, :] = numpy.array([1986, 20, 0, 0, 0, 1, 0, 1, 1, 0])
        vertexValues[3, :] = numpy.array([1986, 20, 0, 0, 0, 1, 0, 2, 2, 0])
        vertexValues[4, :] = numpy.array([1986, 20, 0, 0, 0, 2, 0, 5, 5, 0])
        vertexValues[5, :] = numpy.array([1986, 28, 0, 0, 0, 3, 0, 1, 1, 1])
        vertexValues[6, :] = numpy.array([1986, 26, 1, 0, 1, 3, 6, 1, 1, 1])
        vertexValues[7, :] = numpy.array([1986, 35, 0, 0, 0, 2, 0, 0, 0, 0])
        vertexValues[8, :] = numpy.array([1986, 37, 0, 1, 0, 3, 0, 5, 3, 0])
        vertexValues[9, :] = numpy.array([1986, 40, 0, 1, 0, 4, 0, 3, 3, 0])

        #Check if the values of the vertices are correct 
        for i in range(sparseMultiGraph.getNumVertices()):
            self.assertTrue((sparseMultiGraph.getVertex(i) == vertexValues[i]).all())

        #Now check edges
        edges = numpy.zeros((10, 3))
        edges[0, :] = numpy.array([4, 0, 0])
        edges[1, :] = numpy.array([4, 1, 0])
        edges[2, :] = numpy.array([5, 3, 0])
        edges[3, :] = numpy.array([7, 1, 0])
        edges[4, :] = numpy.array([8, 0, 0])
        edges[5, :] = numpy.array([4, 1, 1])
        edges[6, :] = numpy.array([8, 1, 1])
        edges[7, :] = numpy.array([8, 2, 1])
        edges[8, :] = numpy.array([8, 4, 1])
        edges[9, :] = numpy.array([9, 0, 1])

        self.assertTrue((sparseMultiGraph.getAllEdges() == edges).all())

        #Now test directed graphs
        sparseMultiGraph = multiGraphCsvReader.readGraph(vertexFileName, edgeFileNames, False)

        for i in range(sparseMultiGraph.getNumVertices()):
            self.assertTrue((sparseMultiGraph.getVertex(i) == vertexValues[i]).all())


        edges = numpy.zeros((10, 3))
        edges[0, :] = numpy.array([0, 4, 0])
        edges[1, :] = numpy.array([1, 7, 0])
        edges[2, :] = numpy.array([3, 5, 0])
        edges[3, :] = numpy.array([4, 1, 0])
        edges[4, :] = numpy.array([8, 0, 0])
        edges[5, :] = numpy.array([0, 9, 1])
        edges[6, :] = numpy.array([1, 8, 1])
        edges[7, :] = numpy.array([2, 8, 1])
        edges[8, :] = numpy.array([4, 1, 1])
        edges[9, :] = numpy.array([8, 4, 1])
        
        self.assertTrue((sparseMultiGraph.getAllEdges() == edges).all())

        #Next test out graphs with edge weights
        

if __name__ == '__main__':
    unittest.main()