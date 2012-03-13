

import unittest
import numpy
import logging
import sys 
from apgl.graph.VertexList import VertexList
from apgl.graph.SparseGraph import SparseGraph
from apgl.graph.GraphStatistics import GraphStatistics
from apgl.generator.ErdosRenyiGenerator import ErdosRenyiGenerator

class  GraphStatisticsTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        numpy.set_printoptions(suppress=True, precision=3, linewidth=150)

        #Get an invalid value in substract when computing std in meanSeqScalarStats 
        #numpy.seterr(all='raise')

    def testScalarStatistics(self):
        numFeatures = 1
        numVertices = 10
        vList = VertexList(numVertices, numFeatures)
        graph = SparseGraph(vList)
        graph.addEdge(0, 1)

        growthStatistics = GraphStatistics()
        statsArray = growthStatistics.scalarStatistics(graph)

        #logging.debug(statsArray)

        self.assertTrue(statsArray[growthStatistics.numVerticesIndex] == 10.0)
        self.assertTrue(statsArray[growthStatistics.numEdgesIndex] == 1.0)
        self.assertTrue(statsArray[growthStatistics.maxComponentSizeIndex] == 2.0)
        self.assertTrue(statsArray[growthStatistics.maxComponentEdgesIndex] == 1.0)
        self.assertTrue(statsArray[growthStatistics.numComponentsIndex] == 9.0)
        self.assertEquals(statsArray[growthStatistics.meanComponentSizeIndex], 10.0/9.0)
        self.assertTrue(statsArray[growthStatistics.meanDegreeIndex] == 0.2)
        self.assertTrue(statsArray[growthStatistics.diameterIndex] == 1.0)
        self.assertTrue(statsArray[growthStatistics.effectiveDiameterIndex] == 1.0)
        self.assertTrue(statsArray[growthStatistics.densityIndex] == 1.0/45)
        self.assertEquals(statsArray[growthStatistics.geodesicDistanceIndex], 1.0/55)
        self.assertEquals(statsArray[growthStatistics.harmonicGeoDistanceIndex], 55.0)
        self.assertEquals(statsArray[growthStatistics.geodesicDistMaxCompIndex], 1.0/3)
        self.assertEquals(statsArray[growthStatistics.numNonSingletonComponentsIndex], 1.0)
        self.assertEquals(statsArray[growthStatistics.numTriOrMoreComponentsIndex], 0.0)
        self.assertEquals(statsArray[growthStatistics.secondComponentSizeIndex], 1.0)
        self.assertEquals(statsArray[growthStatistics.maxCompMeanDegreeIndex], 1.0)

        graph.addEdge(0, 2)

        graph.addEdge(3,4)
        graph.addEdge(3,5)
        graph.addEdge(3,6)
        graph.addEdge(7,8)

        statsArray = growthStatistics.scalarStatistics(graph)
        self.assertEquals(statsArray[growthStatistics.numNonSingletonComponentsIndex], 3.0)
        self.assertEquals(statsArray[growthStatistics.numTriOrMoreComponentsIndex], 2.0)
        self.assertEquals(statsArray[growthStatistics.secondComponentSizeIndex], 3.0)
        self.assertEquals(statsArray[growthStatistics.maxCompMeanDegreeIndex], 1.5)

        #Test on a directed graph 
        graph = SparseGraph(vList, False)
        graph.addEdge(0, 1)

        statsArray = growthStatistics.scalarStatistics(graph, treeStats=True)

        self.assertTrue(statsArray[growthStatistics.numVerticesIndex] == 10.0)
        self.assertTrue(statsArray[growthStatistics.numEdgesIndex] == 1.0)
        self.assertTrue(statsArray[growthStatistics.maxComponentSizeIndex] == -1)
        self.assertTrue(statsArray[growthStatistics.maxComponentEdgesIndex] == -1)
        self.assertTrue(statsArray[growthStatistics.numComponentsIndex] == -1)
        self.assertTrue(statsArray[growthStatistics.meanComponentSizeIndex] == -1)
        self.assertEquals(statsArray[growthStatistics.meanDegreeIndex], 0.1)
        self.assertTrue(statsArray[growthStatistics.diameterIndex] == 1.0)
        self.assertTrue(statsArray[growthStatistics.effectiveDiameterIndex] == 1.0)
        self.assertTrue(statsArray[growthStatistics.densityIndex] == 1.0/90)
        self.assertEquals(statsArray[growthStatistics.geodesicDistanceIndex], 1.0/100)
        self.assertEquals(statsArray[growthStatistics.harmonicGeoDistanceIndex], 100)
        self.assertEquals(statsArray[growthStatistics.meanTreeSizeIndex], 10.0/9)
        self.assertEquals(statsArray[growthStatistics.meanTreeDepthIndex], 1.0/9)
        self.assertEquals(statsArray[growthStatistics.maxTreeSizeIndex], 2.0)
        self.assertEquals(statsArray[growthStatistics.maxTreeDepthIndex], 1.0)
        self.assertEquals(statsArray[growthStatistics.numTreesIndex], 9.0)
        self.assertEquals(statsArray[growthStatistics.numNonSingletonTreesIndex], 1.0)
        

        #Test that the max tree is decribed correctly
        graph.addEdge(2, 3)
        graph.addEdge(2, 4)
        graph.addEdge(3, 5)
        graph.addEdge(3, 6)

        statsArray = growthStatistics.scalarStatistics(graph, treeStats=True)
        self.assertEquals(statsArray[growthStatistics.maxTreeSizeIndex], 5.0)
        self.assertEquals(statsArray[growthStatistics.maxTreeDepthIndex], 2.0)
        self.assertEquals(statsArray[growthStatistics.secondTreeSizeIndex], 2.0)
        self.assertEquals(statsArray[growthStatistics.secondTreeDepthIndex], 1.0)
        self.assertEquals(statsArray[growthStatistics.numTreesIndex], 5.0)
        self.assertEquals(statsArray[growthStatistics.numNonSingletonTreesIndex], 2.0)

        #Try a zero size graph
        numFeatures = 0
        numVertices = 0
        vList = VertexList(numVertices, numFeatures)
        graph = SparseGraph(vList)

        statsArray = growthStatistics.scalarStatistics(graph)
        self.assertEquals(statsArray[growthStatistics.numVerticesIndex], 0)
        self.assertEquals(statsArray[growthStatistics.numEdgesIndex], 0)
        self.assertEquals(statsArray[growthStatistics.maxComponentSizeIndex], 0)
        self.assertTrue(statsArray[growthStatistics.maxComponentEdgesIndex] == 0)
        self.assertEquals(statsArray[growthStatistics.numComponentsIndex], 0)
        self.assertEquals(statsArray[growthStatistics.meanComponentSizeIndex], 0)
        self.assertEquals(statsArray[growthStatistics.meanDegreeIndex], 0)
        self.assertEquals(statsArray[growthStatistics.diameterIndex], 0)
        self.assertEquals(statsArray[growthStatistics.effectiveDiameterIndex], 0)
        self.assertEquals(statsArray[growthStatistics.densityIndex], 0)
        self.assertEquals(statsArray[growthStatistics.geodesicDistanceIndex], 0.0)
        self.assertEquals(statsArray[growthStatistics.harmonicGeoDistanceIndex], 0.0)
        self.assertEquals(statsArray[growthStatistics.geodesicDistMaxCompIndex], 0.0)
        self.assertEquals(statsArray[growthStatistics.numNonSingletonComponentsIndex], 0.0)

        graph = SparseGraph(vList, False)

        statsArray = growthStatistics.scalarStatistics(graph)
        self.assertEquals(statsArray[growthStatistics.numVerticesIndex], 0)
        self.assertEquals(statsArray[growthStatistics.numEdgesIndex], 0)
        self.assertEquals(statsArray[growthStatistics.maxComponentSizeIndex], -1)
        self.assertEquals(statsArray[growthStatistics.numComponentsIndex], -1)
        self.assertEquals(statsArray[growthStatistics.meanComponentSizeIndex], -1)
        self.assertEquals(statsArray[growthStatistics.maxComponentEdgesIndex], -1)
        self.assertEquals(statsArray[growthStatistics.meanDegreeIndex], 0)
        self.assertEquals(statsArray[growthStatistics.diameterIndex], 0)
        self.assertEquals(statsArray[growthStatistics.effectiveDiameterIndex], 0)
        self.assertEquals(statsArray[growthStatistics.densityIndex], 0)
        self.assertEquals(statsArray[growthStatistics.geodesicDistanceIndex], 0.0)
        self.assertEquals(statsArray[growthStatistics.harmonicGeoDistanceIndex], 0.0)
        self.assertEquals(statsArray[growthStatistics.geodesicDistMaxCompIndex], -1)
        self.assertEquals(statsArray[growthStatistics.meanTreeSizeIndex], -1)
        self.assertEquals(statsArray[growthStatistics.meanTreeDepthIndex], -1)
        self.assertEquals(statsArray[growthStatistics.numNonSingletonComponentsIndex], -1)
        self.assertEquals(statsArray[growthStatistics.numTreesIndex], -1)
        self.assertEquals(statsArray[growthStatistics.numNonSingletonTreesIndex], -1)


    def testSequenceScalarStats(self):
        numFeatures = 1
        numVertices = 10
        vList = VertexList(numVertices, numFeatures)
        graph = SparseGraph(vList)

        graph.addEdge(0, 2)
        graph.addEdge(0, 1)

        subgraphIndices = [[0, 1, 3], [0, 1, 2, 3]]

        growthStatistics = GraphStatistics()
        statsArray = growthStatistics.sequenceScalarStats(graph, subgraphIndices)

        self.assertTrue(statsArray[0, 0] == 3.0)
        self.assertTrue(statsArray[1, 0] == 4.0)
        self.assertTrue(statsArray[0, 1] == 1.0)
        self.assertTrue(statsArray[1, 1] == 2.0)

    def testVectorStatistics(self):
        numFeatures = 1
        numVertices = 10
        vList = VertexList(numVertices, numFeatures)
        graph = SparseGraph(vList)

        graph.addEdge(0, 2)
        graph.addEdge(0, 1)

        growthStatistics = GraphStatistics()
        statsDict = growthStatistics.vectorStatistics(graph)

        self.assertTrue((statsDict["outDegreeDist"] == numpy.array([7,2,1])).all())
        self.assertTrue((statsDict["inDegreeDist"] == numpy.array([7,2,1])).all())
        self.assertTrue((statsDict["hopCount"] == numpy.array([10,14,16])).all())
        self.assertTrue((statsDict["triangleDist"] == numpy.array([10])).all())

        W = graph.getWeightMatrix()
        W = (W + W.T)/2
        lmbda, V = numpy.linalg.eig(W)
        maxEigVector = V[:, numpy.argmax(lmbda)]
        lmbda = numpy.flipud(numpy.sort(lmbda[lmbda>0]))
        self.assertTrue((statsDict["maxEigVector"] == maxEigVector).all())
        self.assertTrue((statsDict["eigenDist"] == lmbda).all())
        self.assertTrue((statsDict["componentsDist"] == numpy.array([0, 7, 0, 1])).all())

        graph.addEdge(0, 3)
        graph.addEdge(0, 4)
        graph.addEdge(1, 4)

        growthStatistics = GraphStatistics()
        statsDict = growthStatistics.vectorStatistics(graph)

        self.assertTrue((statsDict["outDegreeDist"] == numpy.array([5,2,2,0,1])).all())
        self.assertTrue((statsDict["inDegreeDist"] == numpy.array([5,2,2,0,1])).all())
        self.assertTrue((statsDict["hopCount"] == numpy.array([10,20,30])).all())
        self.assertTrue((statsDict["triangleDist"] == numpy.array([7, 0, 3])).all())

        W = graph.getWeightMatrix()
        W = (W + W.T)/2
        lmbda, V = numpy.linalg.eig(W)
        maxEigVector = V[:, numpy.argmax(lmbda)]
        lmbda = numpy.flipud(numpy.sort(lmbda[lmbda>0]))
        self.assertTrue((statsDict["maxEigVector"] == maxEigVector).all())
        self.assertTrue((statsDict["eigenDist"] == lmbda).all())
        self.assertTrue((statsDict["componentsDist"] == numpy.array([0, 5, 0, 0, 0, 1])).all())

        #Test on a directed graph and generating tree statistics 
        vList = VertexList(numVertices, numFeatures)
        graph = SparseGraph(vList, False)

        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(2, 3)

        graph.addEdge(4, 5)
        
        statsDict = growthStatistics.vectorStatistics(graph, treeStats=True)

        self.assertTrue(( statsDict["inDegreeDist"] == numpy.array([6, 4]) ).all())
        self.assertTrue(( statsDict["outDegreeDist"] == numpy.array([7, 2, 1]) ).all())
        self.assertTrue(( statsDict["triangleDist"] == numpy.array([10]) ).all())
        self.assertTrue(( statsDict["treeSizesDist"] == numpy.array([0, 4, 1, 0, 1]) ).all())
        self.assertTrue(( statsDict["treeDepthsDist"] == numpy.array([4, 1, 1]) ).all())

    def testSequenceVectorStats(self):
        numFeatures = 1
        numVertices = 10
        vList = VertexList(numVertices, numFeatures)
        graph = SparseGraph(vList)

        graph.addEdge(0, 2)
        graph.addEdge(0, 1)

        subgraphIndices = [[0, 1, 3], [0, 1, 2, 3]]

        growthStatistics = GraphStatistics()
        statsList = growthStatistics.sequenceVectorStats(graph, subgraphIndices)

    def testMeanSeqScalarStats(self):
        numFeatures = 1
        numVertices = 10
        vList = VertexList(numVertices, numFeatures)
        
        p = 0.1 
        generator = ErdosRenyiGenerator(p)

        numGraphs = 50
        graphList = []
        subgraphIndices = [list(range(3)), list(range(6)), list(range(10))]

        for i in range(numGraphs): 
            graph = generator.generate(SparseGraph(vList))
            graphList.append((graph, subgraphIndices))

        growthStatistics = GraphStatistics()
        meanStats, stdStats = growthStatistics.meanSeqScalarStats(graphList)

        #Check some of the stats
        for i in range(len(subgraphIndices)):
            self.assertEquals(meanStats[i, growthStatistics.numVerticesIndex], len(subgraphIndices[i]))
            self.assertEquals(stdStats[i, growthStatistics.numVerticesIndex], 0)

            self.assertAlmostEquals(meanStats[i, growthStatistics.numEdgesIndex], 0.5*(len(subgraphIndices[i])*(len(subgraphIndices[i])-1))*p, places=0)

if __name__ == '__main__':
    unittest.main()

