import numpy
import logging
import sys
import scipy.sparse
from apgl.graph import *
from apgl.generator import * 
from apgl.util import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class SparseGraphProfile(object):
    def __init__(self):
        numVertices = 1000
        numFeatures = 0
        vList = VertexList(numVertices, numFeatures)
        W = scipy.sparse.lil_matrix((numVertices, numVertices))
        graph = SparseGraph(vList, W=W)
        p = 0.4
        generator = ErdosRenyiGenerator(p)

        
        self.graph = generator.generate(graph)

    def profileSubgraph(self):
        numVertices = 500
        numFeatures = 0
        vList = VertexList(numVertices, numFeatures)
        graph = SparseGraph(vList)
        generator = ErdosRenyiGenerator(graph)

        p = 0.1

        graph = generator.generateGraph(p)
        indices = numpy.random.permutation(numVertices)

        ProfileUtils.profile('graph.subgraph(indices)', globals(), locals())

    def profileDiameter(self):
        numVertices = 100
        numFeatures = 0
        vList = VertexList(numVertices, numFeatures)
        graph = SparseGraph(vList)
        generator = ErdosRenyiGenerator(graph)

        p = 0.1

        graph = generator.generateGraph(p)

        ProfileUtils.profile('graph.diameter2()', globals(), locals())

    def profileDijkstrasAlgorithm(self):
        n = 10 

        def runDijkstrasAlgorithm():
            for i in range(n):
                self.graph.dijkstrasAlgorithm(i)

        ProfileUtils.profile('runDijkstrasAlgorithm()', globals(), locals())

    def profileFitPowerLaw(self):
        numVertices = 5000
        numFeatures = 0
        vList = VertexList(numVertices, numFeatures)
        graph = SparseGraph(vList, True)

        ell = 2
        m = 2
        generator = BarabasiAlbertGenerator(graph)
        generator.generateGraph(ell, m)
        #print(graph.degreeDistribution())

        alpha, ks, xmin = graph.fitPowerLaw()

        ProfileUtils.profile('graph.fitPowerLaw()', globals(), locals())

    def profileSparseMatrices(self):
        A = numpy.random.random_integers(0, 1, (1000, 1000))
        #W = scipy.sparse.csr_matrix(A)
        W = scipy.sparse.coo_matrix(A)
        #W.sort_indices()

        #Results: lil_matrix has the fastest row operations but very slow column ones
        #csr_matrix has fast-ish row operations and slightly slower column ones
        #csc_matrix is the opposite

        def getRows():
            for i in range(W.shape[0]):
                W.getrow(i).getnnz()

            #W.tocsc()
            #for i in range(W.shape[0]):
            #    W.getcol(i).getnnz()

        ProfileUtils.profile('getRows()', globals(), locals())

    def profileOutDegreeSequence(self):
        ProfileUtils.profile('self.graph.outDegreeSequence()', globals(), locals())

    def profileNeighbours(self):
        def allNeighbours():
            for i in range(self.graph.getNumVertices()):
                self.graph.neighbours(i)

        ProfileUtils.profile('allNeighbours()', globals(), locals())

    def profileAddEdges(self):
        """
        We want to try creating a huge graph and adding lots of edges.
        """
        numVertices = 1000000
        numFeatures = 0
        vList = VertexList(numVertices, numFeatures)
        W = scipy.sparse.lil_matrix((numVertices, numVertices))
        graph = SparseGraph(vList, W=W)

        def runAddEdges():
            numEdges = 100000
            edgesArray = numpy.zeros((numEdges, 2))

            for i in range(numEdges):
                vertexInd1 = numpy.random.randint(0, numVertices)
                vertexInd2 = numpy.random.randint(0, numVertices)
                edgesArray[i, 0] = vertexInd1
                edgesArray[i, 1] = vertexInd2

            graph.addEdges(edgesArray)

        ProfileUtils.profile('runAddEdges()', globals(), locals())
        
    def profileConnectedComponents(self):
        ProfileUtils.profile('self.graph.findConnectedComponents()', globals(), locals())

    def profileFindTrees(self):
        #Need a way to generate random trees 
        pass 

profiler = SparseGraphProfile()
#profiler.profileSubgraph()
#profiler.profileDiameter()
profiler.profileDijkstrasAlgorithm()
#profiler.profileFitPowerLaw()
#profiler.profileSparseMatrices()
#profiler.profileOutDegreeSequence()
#profiler.profileNeighbours()
#profiler.profileAddEdges()