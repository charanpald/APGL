
import unittest
import numpy
import scipy.linalg
import math
import sys 
import logging 
import os.path
import numpy.testing as nptst 
from exp.sandbox.NingSpectralClustering import NingSpectralClustering
from apgl.graph import *
from apgl.generator import *
from apgl.util.Util import Util 

class NingSpectralClusteringTest(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(linewidth=200, threshold=50000, suppress=True)
        numpy.random.seed(21)
        numpy.seterr("raise")
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    def testIncrementEigenSystem(self):
        print "< testIncrementEigenSystem >"
        numVertices = 10
        graph = SparseGraph(GeneralVertexList(numVertices))

        p = 0.4
        generator = ErdosRenyiGenerator(p)
        graph = generator.generate(graph)

        W = graph.getWeightMatrix()
        L = graph.laplacianMatrix()
        degrees = graph.outDegreeSequence()
        D = numpy.diag(degrees)
        
        lmbda1, Q1 = scipy.linalg.eig(L, D)
        lmbda1 = lmbda1.real
        Q1 = Q1.dot(numpy.diag(numpy.diag(Q1.T.dot(D).dot(Q1))**-0.5))

        tol = 10**-6
        k = 3
        inds = numpy.argsort(lmbda1)[0:k]
        lmbda1, Q1 = Util.indEig(lmbda1, Q1, inds)

        #Similarity change vector
        w = graph.getEdge(5,7)
        deltaW = 0.5

        k = 3
        clusterer = NingSpectralClustering(k)
        lmbda2Approx, Q2Approx = clusterer.incrementEigenSystem(lmbda1, Q1, W, 5, 7, deltaW)

        #Compute real eigenvectors then compare against these
        Lhat = L.copy();
        Lhat[5,5] += deltaW; Lhat[7,7] += deltaW
        Lhat[5,7] -= deltaW; Lhat[7,5] -= deltaW
        Dhat = numpy.diag(numpy.diag(Lhat))
        lmbda2, Q2 = scipy.linalg.eig(Lhat, Dhat)
        lmbda2, Q2 = Util.indEig(lmbda2, Q2, inds)

        Q2Approx = Q2Approx.dot(numpy.diag(numpy.diag(Q2Approx.T.dot(Q2Approx))**-0.5))
        Q2 = Q2.dot(numpy.diag(numpy.sum(Q2**2, 0)**-0.5))
        Q1 = Q1.dot(numpy.diag(numpy.sum(Q1**2, 0)**-0.5))

        #Errors in the eigenvalues
        logging.debug("Eigenvalue Errors")
        logging.debug(numpy.linalg.norm(lmbda2 - lmbda2Approx))
        logging.debug(numpy.linalg.norm(lmbda2 - lmbda1))

        #Compute error according to the paper 
        error = numpy.sum(1 - numpy.diag(Q2.T.dot(Q2Approx))**2)
        error2 = numpy.sum(1 - numpy.diag(Q2.T.dot(Q1))**2)
        logging.debug("Eigenvector Errors")
        logging.debug(error)
        logging.debug(error2)

        #Bizarely the eigenvectors are accurate but eigenvalues are worse
        #than those of the original graph. 

    def testIncrementEigenSystem2(self):
        print "< testIncrementEigenSystem2 >"
        """
        We use the example from the paper to see if the error in the eigenvalues
        and eigenvectors decreases. 
        """

        numVertices = 10
        graph = SparseGraph(GeneralVertexList(numVertices))

        graph.addEdge(0, 1, 0.7)
        graph.addEdge(1, 2, 0.4)
        graph.addEdge(2, 3, 0.3)
        graph.addEdge(1, 3, 0.1)
        graph.addEdge(0, 4, 0.5)
        graph.addEdge(3, 4, 0.4)
        graph.addEdge(4, 5, 0.8)
        graph.addEdge(3, 5, 0.3)
        graph.addEdge(6, 5, 0.4)
        graph.addEdge(5, 9, 0.5)
        graph.addEdge(6, 9, 0.3)
        graph.addEdge(6, 7, 0.1)
        graph.addEdge(6, 8, 0.6)
        graph.addEdge(7, 8, 0.7)
        graph.addEdge(9, 8, 0.7)

        W = graph.getWeightMatrix()
        L = graph.laplacianWeightMatrix()
        degrees = numpy.sum(W, 0)
        D = numpy.diag(degrees)

        k = 3 
        lmbda1, Q1 = scipy.linalg.eig(L, D)
        inds = numpy.argsort(lmbda1)[0:k]
        lmbda1, Q1 = Util.indEig(lmbda1, Q1, inds)
        lmbda1 = lmbda1.real 

        #Remove edge 0, 4
        r = numpy.zeros(numVertices, numpy.complex)
        deltaW = -0.5

        clusterer = NingSpectralClustering(k)
        lmbda2Approx, Q2Approx = clusterer.incrementEigenSystem(lmbda1, Q1, W, 0, 4, deltaW)
        
        #Compute real eigenvectors then compare against these
        Lhat = L + numpy.outer(r, r)
        Dhat = numpy.diag(numpy.diag(Lhat))
        lmbda2, Q2 = scipy.linalg.eig(Lhat, Dhat)
        lmbda2, Q2 = Util.indEig(lmbda2, Q2, inds)

        Q2Approx = Q2Approx.dot(numpy.diag(numpy.diag(Q2Approx.T.dot(Q2Approx))**-0.5))
        Q2 = Q2.dot(numpy.diag(numpy.sum(Q2**2, 0)**-0.5))
        Q1 = Q1.dot(numpy.diag(numpy.sum(Q1**2, 0)**-0.5))

        #Compute error according to the paper
        #2 iterations works best - 3 seems to be worse!!! 
        error2 = 1 - numpy.diag(Q2.T.dot(Q2Approx))**2
        errors2 = 1 - numpy.diag(Q2.T.dot(Q1))**2
        logging.debug("Eigenvector Errors")
        logging.debug(error2)
        logging.debug(errors2)

    def testIncrementalEigenSystem3(self):
        print "< testIncrementEigenSystem3 >"
        """
        Test case where we add a vertex and need to increase size of eigenvectors. 
        """
        numVertices = 8
        graph = SparseGraph(GeneralVertexList(numVertices))

        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(1, 2)
        graph.addEdge(3, 4)
        graph.addEdge(3, 5)
        graph.addEdge(4, 5)
        graph.addEdge(0, 3)
        graph.addEdge(1, 6)
        graph.addEdge(4, 7)

        subgraph = graph.subgraph(range(7))

        W1 = subgraph.getWeightMatrix()
        L1 = subgraph.laplacianWeightMatrix()
        degrees1 = numpy.sum(W1, 0)
        D1 = numpy.diag(degrees1)

        W2 = graph.getWeightMatrix()
        L2 = graph.laplacianWeightMatrix()
        degrees1 = numpy.sum(W2, 0)
        D2 = numpy.diag(degrees1)

        k = 3
        lmbda1, Q1 = scipy.linalg.eig(L1, D1)
        inds = numpy.argsort(lmbda1)[0:k]
        lmbda1, Q1 = Util.indEig(lmbda1, Q1, inds)
        lmbda1 = lmbda1.real

        L1hat = numpy.r_[numpy.c_[L1, numpy.zeros(numVertices-1)], numpy.zeros((1, numVertices))]
        W1hat = numpy.r_[numpy.c_[W1, numpy.zeros(numVertices-1)], numpy.zeros((1, numVertices))]
        D1hat = numpy.r_[numpy.c_[D1, numpy.zeros(numVertices-1)], numpy.zeros((1, numVertices))]

        lmbda1, Q2 = scipy.linalg.eig(L2, D2)
        inds = numpy.argsort(lmbda1)[0:k]
        lmbda1, Q2 = Util.indEig(lmbda1, Q2, inds)
        lmbda1 = lmbda1.real
        
        Q1 = numpy.r_[Q1, numpy.ones((1, Q1.shape[1]))]

        #Increase size of eigenvector - not clear how to do this 

        clusterer = NingSpectralClustering(k)
        lmbda2Approx, Q2Approx = clusterer.incrementEigenSystem(lmbda1, Q1, W1hat, 4, 7, 1)

        Q2Approx = Q2Approx.dot(numpy.diag(numpy.diag(Q2Approx.T.dot(Q2Approx))**-0.5))
        Q2 = Q2.dot(numpy.diag(numpy.sum(Q2**2, 0)**-0.5))
        Q1 = Q1.dot(numpy.diag(numpy.sum(Q1**2, 0)**-0.5))

        #Setting the last value of the eigenvectors to zero seems to improve
        #over setting them to 1, but the last eigenvector has a huge error. 
        errors1 = 1 - numpy.diag(Q2.T.dot(Q2Approx))**2
        errors2 = 1 - numpy.diag(Q2.T.dot(Q1))**2
        logging.debug("Eigenvector Errors for added vertex")
        logging.debug(errors1)
        logging.debug(errors2)

    def testCluster(self):
        print "< testCluster >"
        numVertices = 8
        graph = SparseGraph(GeneralVertexList(numVertices))

        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(1, 2)

        graph.addEdge(3, 4)
        graph.addEdge(3, 5)
        graph.addEdge(4, 5)

        graph.addEdge(0, 3)

        W = graph.getWeightMatrix()

        graphIterator = []
        graphIterator.append(W[0:6, 0:6].copy())
        W[1, 6] += 1
        W[6, 1] += 1
        graphIterator.append(W[0:7, 0:7].copy())
        W[4, 7] += 1
        W[7, 4] += 1
        graphIterator.append(W.copy())
        graphIterator = iter(graphIterator)

        k = 2
        clusterer = NingSpectralClustering(k)
        clustersList = clusterer.cluster(graphIterator)

        #Why are the bottom rows of Q still zero?

        #Try example in which only edges change
        numVertices = 7
        graph = SparseGraph(GeneralVertexList(numVertices))

        graph.addEdge(0, 1)
        graph.addEdge(0, 2)
        graph.addEdge(1, 2)

        graph.addEdge(3, 4)

        WList = [] 
        W = graph.getWeightMatrix()
        WList.append(W[0:5, 0:5].copy())

        graph.addEdge(3, 5)
        graph.addEdge(4, 5)
        W = graph.getWeightMatrix()
        WList.append(W[0:6, 0:6].copy())

        graph.addEdge(0, 6)
        graph.addEdge(1, 6)
        graph.addEdge(2, 6)
        W = graph.getWeightMatrix()
        WList.append(W[0:7, 0:7].copy())

        iterator = iter(WList)
        clustersList = clusterer.cluster(iterator)

        #Seems to work, amazingly 
        #print(clustersList)
        
        #Try removing rows/cols
        W2 = W[0:5, 0:5]
        W3 = W[0:4, 0:4]
        WList = [W, W2, W3]
        iterator = iter(WList)
        clustersList = clusterer.cluster(iterator)
        
        nptst.assert_array_equal(clustersList[0][0:5], clustersList[1])
        nptst.assert_array_equal(clustersList[1][0:4], clustersList[2])

    def testDebug(self): 
        if not os.path.isfile("lmbda.npy"):
            print "blop"
            return
        print "< debugging Nings approach >"
        lmbda = numpy.load("lmbda.npy")
        Q = numpy.load("Q.npy")
        W = numpy.load("W.npy")
        i = numpy.load("i.npy")
        j = numpy.load("j.npy")
        deltaW = numpy.load("deltaW.npy")
        
        clusterer = NingSpectralClustering(Q.shape[1])
        clusterer.incrementEigenSystem(lmbda, Q, W, i, j, deltaW)
        print "</ debugging Nings approach >"

if __name__ == '__main__':
    unittest.main()
