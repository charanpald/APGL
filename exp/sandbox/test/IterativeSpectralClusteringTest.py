
from exp.sandbox.IterativeSpectralClustering import IterativeSpectralClustering
from exp.sandbox.GraphIterators import IncreasingSubgraphListIterator
from exp.sandbox.GraphIterators import DatedPurchasesGraphListIterator
from apgl.graph.VertexList import VertexList
from apgl.graph.SparseGraph import SparseGraph
from apgl.graph.GeneralVertexList import GeneralVertexList
from apgl.generator.BarabasiAlbertGenerator import BarabasiAlbertGenerator
import unittest
import numpy
import logging
import sys
import itertools

class IterativeSpectralClusteringTest(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(suppress=True, precision=3, linewidth=200)
        numpy.random.seed(21)
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    def testIncreasingSubgraphListIterator(self):
        #Create a small graph and try the iterator increasing the number of vertices.
        numVertices = 50
        graph = SparseGraph(GeneralVertexList(numVertices))

        ell = 2 
        m = 2 
        generator = BarabasiAlbertGenerator(ell, m)
        graph = generator.generate(graph)

        indices = numpy.random.permutation(numVertices)
        subgraphIndicesList = [indices[0:5], indices]

        graphIterator = IncreasingSubgraphListIterator(graph, subgraphIndicesList)

        #Try a much longer sequence of vertices
        subgraphIndicesList = []
        for i in range(10, numVertices):
            subgraphIndicesList.append(range(i))

        graphIterator = IncreasingSubgraphListIterator(graph, subgraphIndicesList)

        k1 = 3
        k2 = 6
        clusterer = IterativeSpectralClustering(k1, k2)
        clustersList = clusterer.clusterFromIterator(graphIterator)

        #Now test the Nystrom method
        clusterer = IterativeSpectralClustering(k1, k2, alg="nystrom")
        clustersList = clusterer.clusterFromIterator(graphIterator)

    def testClusterOnIncreasingGraphs(self):
        #Create a large graph and try the clustering.
        numClusters = 3
        ClusterSize = 30
        numFeatures = 0
        
        pNoise = 0
        pClust = 1

        numVertices = numClusters*ClusterSize
        vList = GeneralVertexList(numVertices)

        vList = VertexList(numVertices, numFeatures)
        graph = SparseGraph(vList)

#        ell = 2 
#        m = 2 
#        generator = BarabasiAlbertGenerator(ell, m)
#        graph = generator.generate(graph)
        #Generate matrix of probabilities
        W = numpy.ones((numVertices, numVertices))*pNoise
        for i in range(numClusters):
            W[ClusterSize*i:ClusterSize*(i+1), ClusterSize*i:ClusterSize*(i+1)] = pClust
        P = numpy.random.rand(numVertices, numVertices)
        W = numpy.array(P < W, numpy.float)
        upTriInds = numpy.triu_indices(numVertices)
        W[upTriInds] = 0
        W = W + W.T
        graph = SparseGraph(vList)
        graph.setWeightMatrix(W)

        indices = numpy.random.permutation(numVertices)
        subgraphIndicesList = [indices[0:numVertices/2], indices]

        k1 = numClusters
        k2 = 10
        clusterer = IterativeSpectralClustering(k1, k2)
        #Test full computation of eigenvectors
        graphIterator = IncreasingSubgraphListIterator(graph, subgraphIndicesList)
        clustersList = clusterer.clusterFromIterator(graphIterator, False)

        self.assertEquals(len(clustersList), len(subgraphIndicesList))

        for i in range(len(clustersList)):
            clusters = clustersList[i]
            self.assertEquals(len(subgraphIndicesList[i]), len(clusters))
            #print(clusters)

        #Test full computation of eigenvectors with iterator
        graphIterator = IncreasingSubgraphListIterator(graph, subgraphIndicesList)
        clustersList = clusterer.clusterFromIterator(graphIterator, False)

        self.assertEquals(len(clustersList), len(subgraphIndicesList))

        for i in range(len(clustersList)):
            clusters = clustersList[i]
            self.assertEquals(len(subgraphIndicesList[i]), len(clusters))
            #print(clusters)

        #Now test approximation of eigenvectors with iterator
        graphIterator = IncreasingSubgraphListIterator(graph, subgraphIndicesList)
        clustersList2 = clusterer.clusterFromIterator(graphIterator)

        for i in range(len(clustersList2)):
            clusters = clustersList2[i]
            self.assertEquals(len(subgraphIndicesList[i]), len(clusters))
            #print(clusters)

        #Test case where 2 graphs are identical
        subgraphIndicesList = []
        subgraphIndicesList.append(range(graph.getNumVertices()))
        subgraphIndicesList.append(range(graph.getNumVertices()))

        graphIterator = IncreasingSubgraphListIterator(graph, subgraphIndicesList)
        clustersList = clusterer.clusterFromIterator(graphIterator, True)
        #print(clustersList)

    def testClusterOnPurchases(self):
        #Create a list of purchases and cluster it
        numProd = 30
        numUser = 30
        numPurchasesPerDate = 10
        numDate = 10
        
        numPurchase = numPurchasesPerDate * numDate
        listProd = numpy.random.randint(0, numProd, numPurchase)
        listUser = numpy.random.randint(0, numUser, numPurchase)
        # third week is the same as first one
        listProd[numPurchasesPerDate*3:numPurchasesPerDate*4] = listProd[:numPurchasesPerDate]
        listUser[numPurchasesPerDate*3:numPurchasesPerDate*4] = listUser[:numPurchasesPerDate]
        listWeek = range(numDate)*numPurchasesPerDate
        listWeek.sort()
        listYear = [2011]*numPurchase
        purchasesList = list(list(tup) for tup in itertools.izip(listProd, listUser, listWeek, listYear))
#        print purchasesList
        
        k1 = 10
        k2 = 10 
        clusterer = IterativeSpectralClustering(k1, k2)
        #Test full computation of eigenvectors 
        graphIterator = DatedPurchasesGraphListIterator(purchasesList)
        clustersList = clusterer.clusterFromIterator(graphIterator, False)


        for i in range(len(clustersList)):
            clusters = clustersList[i]
#            self.assertEquals(len(subgraphIndicesList[i]), len(clusters))
            print(clusters)

        #Now test approximation of eigenvectors 
        graphIterator = DatedPurchasesGraphListIterator(purchasesList)
        clustersList = clusterer.clusterFromIterator(graphIterator, True)

        for i in range(len(clustersList)):
            clusters = clustersList[i]
#            self.assertEquals(len(subgraphIndicesList[i]), len(clusters))
            #print(clusters)

    def testFindCentroids(self):
        V = numpy.random.rand(10, 3)
        clusters = numpy.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        k1 = 2
        k2 = 2
        clusterer = IterativeSpectralClustering(k1, k2)
        centroids = clusterer.findCentroids(V, clusters)

        centroids2 = numpy.zeros((2, 3))
        centroids2[0, :] = numpy.mean(V[0:5, :], 0)
        centroids2[1, :] = numpy.mean(V[5:, :], 0)

        tol = 10**-6
        self.assertTrue(numpy.linalg.norm(centroids - centroids2) < tol)

if __name__ == '__main__':
#    increasingSubgraphListIteratorTestCase = IterativeSpectralClusteringTest('testIncreasingSubgraphListIterator')
#    unittest.TextTestRunner(verbosity=2).run(increasingSubgraphListIteratorTestCase)
    unittest.main()

