
import numpy
import heapq
import scipy
import logging
import os.path
import tempfile 
import base64 
import shutil 

import apgl
from apgl.util.Util import Util
from apgl.util.Parameter import Parameter
from apgl.graph.AbstractSingleGraph import AbstractSingleGraph
from apgl.graph.VertexList import VertexList
from apgl.graph.GeneralVertexList import GeneralVertexList
from apgl.graph.DictGraph import DictGraph

class AbstractMatrixGraph(AbstractSingleGraph):
    """
    An abstract graph object with single edges between vertices. The edge labels
    are stored in a weight matrix, and vertices are stored in a VertexList 
    """

    def getWeightMatrix(self):
        """
        Returns a numpy array of the weight matrix of this graph.

        :returns:  The weight matrix of this graph. 
        """
        Util.abstract()

    def laplacianMatrix(self, outDegree=True):
        """
        Return the Laplacian matrix of this graph, which is defined as L_{ii} = deg(i)
        L_{ij} = -1 if an edge between i and j, otherwise L_{ij} = 0 . For a directed
        graph one can specify whether to use the out-degree or in-degree.

        :param outDegree: whether to use the out-degree for the computation of the degree matrix
        :type outDegree: :class:`bool`

        :returns:  A laplacian adjacency matrix as numpy array.
        """
        A = self.adjacencyMatrix()

        if outDegree:
            D = numpy.diag(self.outDegreeSequence())
        else:
            D = numpy.diag(self.inDegreeSequence())

        return -A + D

    def laplacianWeightMatrix(self, outDegree=True):
        """
        Return the Laplacian matrix of this graph, L = D - W, where D is the degree
        matrix and W is the weight matrix. For a directed graph one can specify whether
        to use the out-degree or in-degree.

        :param outDegree: whether to use the out-degree for the computation of the degree matrix
        :type outDegree: :class:`bool`

        :returns:  A laplacian weight matrix.
        """
        W = self.getWeightMatrix()

        if outDegree:
            D = numpy.diag(numpy.sum(W, 1))
        else:
            D = numpy.diag(numpy.sum(W, 0))

        return D - W

    def normalisedLaplacianSym(self, outDegree=True):
        """
        Compute the normalised symmetric laplacian matrix using L = I - D^-1/2 W D^-1/2,
        in which W is the weight matrix and D_ii is the sum of the ith vertices weights.

        :param outDegree: whether to use the out-degree for the computation of the degree matrix
        :type outDegree: :class:`bool`

        :returns:  A normalised symmetric laplacian matrix as a numpy array.
        """
        W = self.getWeightMatrix()

        if outDegree:
            degrees = numpy.sum(W, 1)
        else:
            degrees = numpy.sum(W, 0)

        D2 = numpy.diag((degrees + (degrees==0))**-0.5)
        
        L = numpy.eye(self.getNumVertices()) - numpy.dot(D2, numpy.dot(W, D2))
        return L

    def normalisedLaplacianRw(self, outDegree=True):
        """
        Compute the normalised random walk laplacian matrix with L = I - D^-1 W in
        which W is the weight matrix and D_ii is the sum of the ith vertices weights.

        :param outDegree: whether to use the out-degree for the computation of the degree matrix
        :type outDegree: :class:`bool`

        :returns:  A normalised random-walk laplacian matrix as a numpy array..
        """
        W = self.getWeightMatrix()
        if outDegree:
            degrees = numpy.sum(W, 1)
        else:
            degrees = numpy.sum(W, 0)

        D2 = numpy.diag((degrees + (degrees==0))**-1)

        L = numpy.eye(self.getNumVertices()) - numpy.dot(D2, W)
        return L


    def getAllVertexIds(self):
        """
        Returns a list of all the vertex IDs of this graph. 
        """
        return list(range(0, self.vList.getNumVertices()))

    def getVertexList(self):
        """
        :returns: the AbstractVertexList object of this graph.
        """
        return self.vList

    def getVertex(self, vertexIndex):
        """
        Returns the vertex associated with the given vertex index.

        :param vertexIndex: the index of the vertex.
        :type vertexIndex: :class:`int`

        :returns:  The value of the vertex at the given index. 
        """
        #The vertexlist should check the parameter (no point doing it twice) 
        return self.vList.getVertex(vertexIndex)

    def setVertex(self, vertexIndex, vertex):
        """
        Set the vertex with given index to a particular value. 

        :param vertexIndex: the index of the vertex.
        :type vertexIndex: :class:`int`

        :param vertex: the value of the vertex.
        """
        Parameter.checkIndex(vertexIndex, 0, self.vList.getNumVertices())
        self.vList.setVertex(vertexIndex, vertex)

    def getNumVertices(self):
        """
        :returns: the number of vertices in this graph.
        """
        return self.vList.getNumVertices()

    def addEdge(self, vertexIndex1, vertexIndex2, edge=1):
        """
        Add a non-zero edge between two vertices.

        :param vertexIndex1: The index of the first vertex.
        :type vertexIndex1: :class:`int`

        :param vertexIndex2: The index of the second vertex.
        :type vertexIndex2: :class:`int`

        :param edge: The value of the edge.
        :type edge: :class:`float`
        """
        Parameter.checkIndex(vertexIndex1, 0, self.vList.getNumVertices())
        Parameter.checkIndex(vertexIndex2, 0, self.vList.getNumVertices())

        if edge == 0 or edge == float('inf'):
            raise ValueError("Cannot add a zero or infinite edge")

        self.W[vertexIndex1, vertexIndex2] = edge
        if self.undirected:
            self.W[vertexIndex2, vertexIndex1] = edge

    def addEdges(self, edgeIndexArray, edgeValues=[]):
        """
        Takes a numpy array of edge index pairs, and edge values and adds them
        to this graph. The array is 2 dimensional such that each row is a pair
        of edge indices.

        :param edgeIndexArray: The array of edge indices with each being a pair of indices.
        :type edgeIndexArray: :class:`numpy.ndarray`

        :param edgeValues: The list of edge values
        :type edgeValues: :class:`list`
        """
        if (edgeIndexArray < 0).any() or (edgeIndexArray >= self.vList.getNumVertices()).any():
            raise ValueError("Invalid indices for edges.")

        if edgeValues == []: 
            edgeValues = numpy.ones(edgeIndexArray.shape[0])
        elif (edgeValues == 0).any():
            raise ValueError("Invalid entry, found zero edge value(s): " + str(numpy.nonzero(edgeValues==0)[0]))

        if self.undirected:
            for i in range(edgeIndexArray.shape[0]):
                self.W[int(edgeIndexArray[i, 0]), int(edgeIndexArray[i, 1])] = edgeValues[i]
                self.W[int(edgeIndexArray[i, 1]), int(edgeIndexArray[i, 0])] = edgeValues[i]
        else:
            for i in range(edgeIndexArray.shape[0]):
                self.W[int(edgeIndexArray[i, 0]), int(edgeIndexArray[i, 1])] = edgeValues[i]


    def getEdge(self, vertexIndex1, vertexIndex2):
        """
        Get the value of an edge, or None if no edge exists. 

        :param vertexIndex1: The index of the first vertex.
        :type vertexIndex1: :class:`int`

        :param vertexIndex2: The index of the second vertex.
        :type vertexIndex2: :class:`int`

        :returns:  The value of the edge between the given vertex indices.
        """
        Parameter.checkIndex(vertexIndex1, 0, self.vList.getNumVertices())
        Parameter.checkIndex(vertexIndex2, 0, self.vList.getNumVertices())

        if self.W[vertexIndex1, vertexIndex2]==0:
            return None
        else:
            return self.W[vertexIndex1, vertexIndex2]

    def getEdgeValues(self, edgeArray):
        """
        Take an array of n x 2 of vertex indices and return the corresponding edge
        values.

        :param edgeArray: An array with an edge on each row
        :type edgeArray: :class:`numpy.ndarray`

        :returns: A vector of n values corresponding to the edge weights of edgeArray
        """
        edgeValues = numpy.zeros(edgeArray.shape[0])

        for i in range(edgeValues.shape[0]):
            edgeValues[i] = self.getEdge(edgeArray[i, 0], edgeArray[i, 1])

        return edgeValues 

    def removeEdge(self, vertexIndex1, vertexIndex2):
        """
        Remove an edge between two vertices.

        :param vertexIndex1: The index of the first vertex.
        :type vertexIndex1: :class:`int`

        :param vertexIndex2: The index of the second vertex.
        :type vertexIndex2: :class:`int`
        """
        Parameter.checkIndex(vertexIndex1, 0, self.vList.getNumVertices())
        Parameter.checkIndex(vertexIndex2, 0, self.vList.getNumVertices())
        self.W[vertexIndex1, vertexIndex2] = 0
        if self.undirected:
            self.W[vertexIndex2, vertexIndex1] = 0

    def isUndirected(self):
        """
        Returns true if this graph is undirected otherwise false. 
        """
        return self.undirected


    def removeAllEdges(self):
        """
        Removes all edges from this graph. 
        """
        self.W = self.W*0

    def __str__(self):
        output = str(self.__class__.__name__) + ": "
        output += "vertices " + str(self.getNumVertices()) + ", edges " + str(self.getNumEdges())
        if self.undirected:
            output += ", undirected"
        else:
            output += ", directed"
        output += ", vertex list type: " +  str(self.vList.__class__.__name__)
        return output


    def diameter(self, useWeights=False, P=None):
        """
        Finds the diameter of a graph i.e. the longest shortest path. If useWeights
        is True then the weights in the adjacency matrix are used if P is not
        provided. 

        :param useWeights: Whether to use edge weights to compute a diameter. 
        :type useWeights: :class:`bool`

        :param P: An optional nxn matrix whose ijth entry is the shortest path from i to j.
        :type P: :class:`ndarray`

        :returns:  The diameter of this graph. 
        """
        Parameter.checkBoolean(useWeights)
        if P!=None and (type(P) != numpy.ndarray or P.shape != (self.getNumVertices(), self.getNumVertices())):
            logging.debug("P.shape = " + P.shape + " W.shape = " + str(self.W.shape))
            raise ValueError("P must be array of same size as weight matrix of graph")
        
        if self.getNumEdges() == 0: 
            return 0 

        if P == None:
            P = self.floydWarshall(useWeights)
        else:
            P = P.copy()

        if useWeights == False:
            return int(numpy.max(P[P!=float('inf')]))
        else:
            return float(numpy.max(P[P!=float('inf')]))

    def effectiveDiameter(self, q, P=None):
        """
        The effective diameter is the minimum d such that for a fraction q of
        reachable node pairs, the path length is at most d. This is more rubust
        than the standard diameter method. One can optionally pass in a matrix
        P whose ijth entry is the shortest path from i to j. 

        :param q: The fraction of node pairs to consider.
        :type q: :class:`float`

        :param P: An optional nxn matrix whose ijth entry is the shortest path from i to j.
        :type P: :class:`ndarray`

        :returns:  The effective diameter of this graph. 
        """
        Parameter.checkFloat(q, 0.0, 1.0)
        if P!=None and (type(P) != numpy.ndarray or P.shape != (self.getNumVertices(), self.getNumVertices())):
            raise ValueError("P must be array of same size as weight matrix of graph")

        if self.getNumEdges() == 0:
            return 0

        #Paths from a vertex to itself are ignored 
        if P == None:
            P = self.floydWarshall(False)
        else:
            P = P.copy()
        
        P[numpy.diag_indices(P.shape[0])] = float('inf')

        paths = numpy.sort(P[P!=float('inf')])
        
        if paths.shape[0] != 0:
            ind = numpy.floor((paths.shape[0]-1)*q)
            return int(paths[ind])
        else:
            return 0.0 
        

    def floydWarshall(self, useWeights=True):
        """
        Use the Floyd-Warshall algorithm to find the shortest path between all pairs
        of vertices. If useWeights is true, then the weights are used to compute the
        path, otherwise adjacencies are used. Note that the shortest path of a
        vertex to itself is always zero. Returns a matrix whose ij th entry is the
        shortest path between vertices i and j. This algorithm scales as O(n^3)
        with the number of vertices n, and is not recommended for very large graphs.

        :param useWeights: Whether to use the edge weight to compute path cost. 
        :type useWeights: :class:`bool`

        :returns:  A matrix of shortest paths between all vertices. 
        """
        numVertices = self.vList.getNumVertices()
        
        if useWeights:
            P = self.getWeightMatrix().copy()
        else:
            P = self.adjacencyMatrix().copy()

        P = P.astype(numpy.float64)
        P[P==0] = float('inf')
        
        j = numpy.ones(numVertices, numpy.int)

        if self.undirected:
            #p = P[numpy.triu_indices(P.shape[0])]

            for k in range(0, numVertices):
                #Q = numpy.outer(P[:, k], j)
                Q = numpy.repeat(numpy.array([P[:, k]]), P.shape[0], 0)
                P2 = Q + Q.T
                P = numpy.minimum(P, P2)
        else:
            for k in range(0, numVertices):
                P2 = numpy.outer(P[:, k], j) + numpy.outer(j, P[k, :])
                P = numpy.minimum(P, P2)

        P[numpy.diag_indices(P.shape[0])] = 0 

        return P

    def maxProductPaths(self):
        """
        Find the maximum product paths between all pairs of vertices using
        a modified version of the Floyd-Warshall algorithm.

        :returns: A matrix P whose ijth entry corresponds to the maximal product of edge weights between them.
        """
        numVertices = self.vList.getNumVertices()
        P = self.getWeightMatrix().copy()
        stepSize = min(100, numVertices-1)

        for k in range(0, numVertices):
            Util.printIteration(k, stepSize, numVertices)
            P2 = numpy.outer(P[:, k], P[k, :])
            P = numpy.maximum(P, P2)

        return P

    def geodesicDistance(self, P=None, vertexInds=None):
        """
        Compute the mean geodesic distance for a graph. This is denoted for an 
        undirected graph by 1/(1/2 n(n+1)) \sum_{i<=j} d_ij where d_ij is the
        shortest path length between i and j. Note that if i and j are not connected
        we assume a path length of 0. If the graph is directed then the geodesic
        distance is 1/(n^2) sum_{i, j} d_ij.

        :param P: An optional nxn matrix whose ijth entry is the shortest path from i to j.
        :type P: :class:`ndarray`

        :param vertexInds: An optional list of vertices used to compute the mean geodesic distance. If this list is none, then all vertices are used.
        :type vertexInds: :class:`list`

        :returns:  The mean geodesic distance of this graph.
        """
        if P!=None and (type(P) != numpy.ndarray or P.shape != (self.getNumVertices(), self.getNumVertices())):
            raise ValueError("P must be array of same size as weight matrix of graph")
        if vertexInds!=None:
            Parameter.checkList(vertexInds, Parameter.checkInt, [0, self.getNumVertices()])
        if self.getNumVertices() == 0 or (vertexInds != None and len(vertexInds)==0):
            return 0
        
        if P == None:
            P = self.floydWarshall(True)
        else:
            P = P.copy()

        if vertexInds != None:
            P = P[vertexInds, :][:, vertexInds]

        n = P.shape[0]
        P[P==numpy.inf] = 0
        P[numpy.diag_indices(P.shape[0])] = 0.0

        distanceSum = numpy.sum(P)
        if self.isUndirected():
            return distanceSum/(n*(n+1))
        else:
            return distanceSum/(n**2)

    def harmonicGeodesicDistance(self, P=None, vertexInds=None):
        """
        Compute the "harmonic mean" geodesic distance for a graph. This is
        denoted by the inverse of 1/(1/2 n(n+1)) \sum_{i<=j} d_ij^-1 where d_ij is the
        shortest path length between i and j for an undirected graph. The distance from a
        node to itself is infinite. For a directed graph, the inverse distance is
        1/n^2 sum_{i,j} d_ij^-1.

        :param P: An optional nxn matrix whose ijth entry is the shortest path from i to j.
        :type P: :class:`ndarray`

        :param vertexInds: An optional list of vertices used to compute the mean geodesic distance. If this list is none, then all vertices are used.
        :type vertexInds: :class:`list`

        :returns:  The mean harmonic geodesic distance of this graph. 
        """
        if P!=None and (type(P) != numpy.ndarray or P.shape != (self.getNumVertices(), self.getNumVertices())):
            raise ValueError("P must be array of same size as weight matrix of graph")
        if vertexInds!=None:
            Parameter.checkList(vertexInds, Parameter.checkInt, [0, self.getNumVertices()])
        if self.getNumVertices() == 0 or (vertexInds != None and len(vertexInds)==0):
            return 0

        if P == None:
            P = self.floydWarshall(True)
        else:
            P = P.copy()

        if vertexInds != None:
            P = P[vertexInds, :][:, vertexInds]

        n = P.shape[0]
        P = 1/(P + numpy.diag(numpy.ones(n)*numpy.inf))

        if self.isUndirected(): 
            distanceSum = numpy.sum(numpy.triu(P))
            if distanceSum != 0:
                return (n*(n+1))/(2*distanceSum)
        else: 
            distanceSum = numpy.sum(P)
            if distanceSum != 0: 
                return n**2/distanceSum

        #Means that all vertices are disconnected 
        return float('inf')

    def adjacencyMatrix(self):
        """
        Return the adjacency matrix in numpy.ndarray format. Warning: should not be used
        unless sufficient memory is available to store the dense matrix.

        :returns: The adjacency matrix in dense format
        """
        W2 = self.getWeightMatrix().copy()
        W2[W2.nonzero()] = 1
        return W2

    def maybeIsomorphicWith(self, graph):
        """
        Returns false if graph is definitely not isomorphic with the current graph,
        however a True may mean the graphs are not isomorphic. Makes a comparison
        with the eigenvalues of the Laplacian matrices.

        :returns: True if the current graph is maybe isomorphic with the input one.
        """
        L1 = self.laplacianMatrix()
        L2 = graph.laplacianMatrix()

        sigma1, V1 = numpy.linalg.eig(L1)
        sigma2, V2 = numpy.linalg.eig(L2)

        sigma1 = numpy.sort(sigma1)
        sigma2 = numpy.sort(sigma2)

        tol = 10**-6

        return numpy.linalg.norm(sigma1 - sigma2) <= tol

    def complement(self):
        """
        Returns a graph with identical vertices (same reference) to the current one, but with the
        complement of the set of edges. Edges that do not exist have weight 1.
        """

        Util.abstract()

    def save(self, filename):
        """
        Save the graph object to the corresponding filename under the .zip extension. The
        adjacency matrix is stored in matrix market format and the AbstractVertexList
        decides how to store the vertex labels. 

        :param filename: The name of the file to save.
        :type filename: :class:`str`

        :returns: The name of the saved zip file.
        """
        Parameter.checkClass(filename, str)
        import zipfile
        
        (path, filename) = os.path.split(filename)
        if path == "":
            path = "./"        
        
        tempPath = tempfile.mkdtemp()

        originalPath = os.getcwd()
        try:
            os.chdir(tempPath)

            self.saveMatrix(self.W, self._wFilename)
            vListFilename = self.vList.save(self._verticesFilename)

            metaDict = {}
            metaDict["version"] = apgl.__version__
            metaDict["undirected"] = self.undirected
            metaDict["vListType"] = self.vList.__class__.__name__
            Util.savePickle(metaDict, self._metaFilename)

            myzip = zipfile.ZipFile(filename + '.zip', 'w')
            myzip.write(self._wFilename)
            myzip.write(vListFilename)
            myzip.write(self._metaFilename)
            myzip.close()

            os.remove(self._wFilename)
            os.remove(vListFilename)
            os.remove(self._metaFilename)
            
            shutil.move(filename + ".zip", path + "/" + filename + '.zip')
        finally:
            os.chdir(originalPath)
            
        os.rmdir(tempPath)
            
        return path + "/" + filename + '.zip'

    @classmethod
    def load(cls, filename):
        """
        Load the graph object from the corresponding file. Data is loaded in a zip
        format as created using save().

        :param filename: The name of the file to load.
        :type filename: :class:`str`

        :returns: A graph corresponding to the one saved in filename.
        """
        Parameter.checkClass(filename, str)
        import zipfile 

        (path, filename) = os.path.split(filename)
        if path == "":
            path = "./"
        
        tempPath = tempfile.mkdtemp()
        originalPath = os.getcwd()
        
        try:
            os.chdir(path)

            myzip = zipfile.ZipFile(filename + '.zip', 'r')
            myzip.extractall(tempPath)
            myzip.close()

            os.chdir(tempPath)

            #Deal with legacy files 
            try:
                W = cls.loadMatrix(cls._wFilename)
                metaDict = Util.loadPickle(cls._metaFilename)
                vList = globals()[metaDict["vListType"]].load(cls._verticesFilename)
                undirected = metaDict["undirected"]

            except IOError:
                W = cls.loadMatrix(filename + cls._matExt)
                vList = VertexList.load(filename)
                undirected = Util.loadPickle(filename + cls._boolExt)

            graph = cls(vList, undirected)
            graph.W = W

            for tempFile in myzip.namelist():
                os.remove(tempFile)
        finally:
            os.chdir(originalPath)

        os.rmdir(tempPath)

        return graph

    def setVertexList(self, vList):
        """
        Assign a new VertexList object to this graph. The number of vertices in the
        VertexList must be the same as in the graph.

        :param vList: A new subclass of AbstractVertexList to assign to this graph. 
        :type vList: :class:`apgl.graph.AbstractVertexList`
        """
        Parameter.checkClass(vList, VertexList)

        if vList.getNumVertices() != self.vList.getNumVertices():
            raise ValueError("Can only set to a VertexList with same number of vertices.")

        self.vList = vList 

    def setWeightMatrix(self, W):
        """
        Set the weight matrix of this graph. Requires as input an ndarray with the
        same dimensions as the current weight matrix. Edges are represented by
        non-zero values.

        :param W: The weight matrix.
        :type W: :class:`ndarray`
        """
        Parameter.checkClass(W, numpy.ndarray)

        if W.shape != (self.vList.getNumVertices(), self.vList.getNumVertices()):
            raise ValueError("Weight matrix has wrong shape : " + str(W.shape))

        if self.undirected and (W != W.T).any():
            raise ValueError("Weight matrix of undirected graph must be symmetric")


        self.W = W 

    def outDegreeSequence(self):
        """
        Return a vector of the (out)degree for each vertex.
        """
        Util.abstract()

    def inDegreeSequence(self):
        """
        :returns: a vector of the (in)degree for each vertex.
        """
        Util.abstract()

    def degreeSequence(self):
        """
        :returns: a vector of the degrees (including self edges) for each vertex for an undirected graph.
        """
        if not self.isUndirected():
            raise ValueError("degreeSequence is only for undirected graphs")

        degSequence = self.outDegreeSequence()

        #A very slow method of adding diagonal entries
        for i in range(self.getNumVertices()):
            if self.getEdge(i, i) != None:
                degSequence[i] += 1

        return degSequence 

    def degreeDistribution(self):
        """
        Return a vector of (out)degree distributions. The ith element of the vector
        corresponds to the frequency of degree i.

        :returns: A vector of (out)degree distributions.
        """
        if self.getNumVertices() == 0 :
            return numpy.array([], numpy.int)

        degrees = self.outDegreeSequence()
        binCounts = numpy.bincount(degrees)
        return binCounts

    def inDegreeDistribution(self):
        """
        Returns a vector of in-degree distributions. The ith element of the vector
        corresponds to the frequency of degree i.

        :returns: A vector of (in)degree distributions.
        """
        if self.getNumVertices() == 0 :
            return numpy.array([], numpy.int)

        degrees = self.inDegreeSequence()
        binCounts = numpy.bincount(degrees)
        return binCounts

    def fitPowerLaw(self):
        """
        Fits the out-degree probabilities of this graph using the power law
        p_d ~ d^-gamma. The value of xmin is the point to start taking examples.

        :returns alpha: The power law exponent.
        :returns ks: A fit of the power law curve to the data using KS.
        :returns xmin: The minimum value of x.
        """
        if self.getNumEdges() == 0:
            return 0,0,0

        logging.warn("Use with caution: fitPowerLaw may not be stable")
        degreeSeq = self.outDegreeSequence()
        degreeMax = numpy.max(degreeSeq)
        xmins = numpy.arange(1, numpy.minimum(20, degreeMax))
        ks, alpha, xmin = Util.fitDiscretePowerLaw(degreeSeq, xmins)

        return alpha, ks, xmin

    def hopCount(self, P=None):
        """
        Returns an array such that the ith element is the number of pairs of
        vertices reachable within i hops. This includes self pairs, and all
        other pairs are counted twice in the undirected case otherwise once.

        :param P: An optional nxn matrix whose ijth entry is the shortest unweighted path from i to j.
        :type P: :class:`ndarray`

        :returns: An array of hop counts. 
        """
        if self.getNumVertices() == 0:
            return numpy.array([])

        if P!=None and (type(P) != numpy.ndarray or P.shape != (self.getNumVertices(), self.getNumVertices())):
            logging.debug("P.shape = " + P.shape + " W.shape = " + str(self.W.shape))
            raise ValueError("P must be array of same size as weight matrix of graph")

        if P == None:
            P = self.floydWarshall(False)
        else:
            P = P.copy()

        p = P.ravel()
        p = p[numpy.logical_not(numpy.isinf(p))]
        p = numpy.array(p, numpy.int32)
        hopCount = numpy.bincount(p)

        return numpy.cumsum(hopCount)

    def triangleSequence(self):
        """
        Computes the number of triangles each vertex participates in using the
        diagonal of the adjcancy matrix. In an undirected graph, a each triangle
        is counted twice (once for each direction). Note that self loops are not
        used to form triangles.

        :returns: An array of triangle counts for each vertex. 
        """

        A = self.adjacencyMatrix()
        A[numpy.diag_indices(A.shape[0])] = 0 
        A3 = numpy.linalg.matrix_power(A, 3)

        return numpy.array(numpy.diag(A3), numpy.int)

    def maxEigenvector(self):
        """
        Returns the eigenvector of maximum eigenvalue of the adjacency matrix. The
        eigenvector is of unit length, and measures the centrality of the corresponding
        vertex. It is based on the principle that connections to high-scoring nodes
        contribute more to the score of the node in question than equal connections
        to low-scoring nodes.

        :returns: The maximum eigenvector of the adjacency matrix. 
        """
        
        A = self.getWeightMatrix()
        w, V = numpy.linalg.eig(A)

        i = numpy.argmax(w)
        return V[:, i]

    def betweenness(self):
        """
        Return the betweenness of each vertex in the graph. The betweenness is
        defined as the number of shortest paths passing through each vertex.

        :returns: A vector of betweenness values of the same length as the number of vertices in the graph.
        """
        n = self.getNumVertices() 
        P = self.adjacencyMatrix().copy()
        P[P==0] = float('inf')
        N = numpy.ones((n, n), numpy.int32)*-1
        pathFreqs = numpy.zeros(n)

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if P[j, i] + P[i, k] <= P[j, k]:
                        P[j, k] = P[j,i]+P[i,k]
                        N[j, k] = i

        #How do we store next values for multiple shortest paths?
        N2 = N[N!=-1]
        return numpy.bincount(N2.ravel())

    def clusteringCoefficient(self):
        """
        Find the global clustering coefficient of this graph as defined here
        http://en.wikipedia.org/wiki/Clustering_coefficient

        :returns: The clustering coefficient of this graph. 
        """
        A = self.adjacencyMatrix()
        A2 = A.dot(A)
        A3 = A2.dot(A)

        diagSumA2 = 0

        for i in range(0, A2.shape[0]):
            diagSumA2 = diagSumA2 + A2[i, i]

        numTriples = A2.sum()-diagSumA2

        diagSumA3 = 0

        for i in range(0, A3.shape[0]):
            diagSumA3 = diagSumA3 + A3[i, i]

        if numTriples == 0:
            return 0
        else:
            return diagSumA3/numTriples

    def adjacencyList(self, useWeights=True):
        """
        Returns an adjacency list representation L of the graph, in which L[i]
        is the list of all neighbours of vertex i. Furthermore, the method 
        returns W in which W[i] which is the corresponding set of weights.

        :param useWeights: If true uses weights of edges as opposed to just adjacencies.
        :type useWeights: :class:`bool`

        :returns L: A list whose ith element is a list of neighbours for vertex i.
        :returns W: A list whose ith element is a list of neighbour weights/adjacencies for vertex i.
        """
        neighbourIndices = []
        neighbourWeights = []

        if useWeights:
            A = self.getWeightMatrix()
        else:
            A = self.adjacencyMatrix()
        
        for i in range(self.getNumVertices()):
            if scipy.sparse.issparse(A):
                Arow = A[i, :].todense()
                Arow = numpy.ravel(Arow)
            else:
                Arow = A[i, :]

            neighbourIndices.append(numpy.nonzero(Arow)[0])
            neighbourWeights.append(Arow[numpy.nonzero(Arow)[0]])


        return neighbourIndices, neighbourWeights


    def dijkstrasAlgorithm(self, vertexIndex, neighbourLists=None):
        """
        Run Dijkstras Algorithm on the graph for a given source vertex. The parameter
        neighbourLists is a tuple containing two lists. The first of this lists contains
        at the ith position all the neighbours of vertex i. The second list contains
        the corresponding weight on the edge. If neighbourLists=None, then it is
        computed automatically and all edge weights are set to 1. Returns
        an array with the distance to all vertices (including itself).

        :param vertexIndex: the index of the source vertex.
        :type vertexIndex: :class:`int`

        :param neighbourLists: A tuple of two lists containing vertex adjacencies and edge weights respectively.
        :type neighbourLists: :class:`list`

        :returns: An array whose ith element is the distance to vertex i. 
        """
        Parameter.checkIndex(vertexIndex, 0, self.getNumVertices())
        if neighbourLists!=None:
            neighbourIndices, neighbourWeights = neighbourLists

            if len(neighbourIndices) != self.getNumVertices() or len(neighbourWeights) != self.getNumVertices():
                raise ValueError("Adjacency lists must be of same size as graph")
        else:
            neighbourIndices, neighbourWeights = self.adjacencyList(False)

        previous = numpy.zeros(self.getNumVertices())
        distance = numpy.ones((self.getNumVertices(), 2))*numpy.inf
        distance[vertexIndex, 0] = 0
        distance[:, 1] = numpy.arange(self.getNumVertices())
        distance = distance.tolist()
        heapq.heapify(distance)

        #Dictionary of the tuples indexed by the vertex index
        distanceDict = {}
        for i in distance:
            distanceDict[i[1]] = i
        INVALID = -1

        distanceArray = numpy.ones(self.getNumVertices())*numpy.inf
        notVisited = numpy.ones(self.getNumVertices(), numpy.bool)

        while len(distanceDict) != 0:
            minVertexIndex = INVALID
            while minVertexIndex == INVALID:
                (minVertexDistance, minVertexIndex) = heapq.heappop(distance)

            distanceArray[minVertexIndex] = minVertexDistance
            del(distanceDict[minVertexIndex])
            notVisited[minVertexIndex] = False 
            if  minVertexDistance == numpy.inf:
                break

            minVertexIndex = int(minVertexIndex)
            cols = neighbourIndices[minVertexIndex]
            weights = neighbourWeights[minVertexIndex]
            #updateDistances(cols, weights, minVertexDistance, distanceDict, previous, distanceArray)

            newDistances = weights + minVertexDistance
            isBetter = numpy.logical_and(newDistances < distanceArray[cols], notVisited[cols])

            for i in range(cols[isBetter].shape[0]):
                j = cols[isBetter][i]
                distanceDict[j][1] = INVALID
                distanceDict[j] = [newDistances[isBetter][i], j]
                heapq.heappush(distance, distanceDict[j])
                distanceArray[j] = newDistances[isBetter][i]

        return distanceArray

    def findAllDistances(self, useWeights=True):
        """
        Use the repeated calls to Dijkstra'  algorithm to find the shortest path between all pairs
        of vertices.  If useWeights is true, then the weights are used to compute the
        path, otherwise adjacencies are used. Note that the shortest path of a
        vertex to itself is always zero. Returns a matrix whose ij th entry is the
        shortest path between vertices i and j.

        :param useWeights: Whether to use the edge weight to compute path cost.
        :type useWeights: :class:`bool`

        :returns:  A matrix of shortest paths between all vertices.
        """

        neighbourLists = self.adjacencyList(useWeights)
        P = numpy.zeros((self.getNumVertices(), self.getNumVertices()))

        for i in range(self.getNumVertices()):
            P[i, :] = self.dijkstrasAlgorithm(i, neighbourLists)

        return P 

    def diameter2(self):
        """
        Use Dijkstras Algorithm to compute the diameter of the graph.

        :returns: The diameter of the graph. 
        """

        maxDiameter = 0

        for i in range(self.getNumVertices()):
            distances = self.dijkstrasAlgorithm(i)
            maxDistance = numpy.max(distances[distances!=numpy.inf])
            if maxDistance > maxDiameter:
                maxDiameter = maxDistance

        return maxDiameter
    
    def egoGraph(self, vertexIndex):
        """
        Returns the subgraph composed of the given vertex and its immediate neighbours.
        In the new graph, the ego is index 0 and neighbours are indexed in order
        after 0.

        :param vertexIndex: the index of the source vertex.
        :type vertexIndex: :class:`int`

        :returns: A subgraph of the current one consisting of only immediate neighbours. 
        """
        Parameter.checkIndex(vertexIndex, 0, self.vList.getNumVertices())

        neighbours = self.neighbours(vertexIndex)
        egoGraphIndices = numpy.r_[numpy.array([vertexIndex]), neighbours]
        return self.subgraph(egoGraphIndices)

    def isTree(self):
        """
        Returns true if this graph is a tree. Every vertex must have an in-degree
        of 1 (i.e. one parent), except the root which has an in-degree of zero
        and non-zero out-degree.

        :returns: A boolean indicating whether the current graph is a tree. 
        """
        if self.isUndirected():
            raise ValueError("Only directed graphs can be trees")

        if self.getNumVertices()==0:
            return True 

        inDegSeq = self.inDegreeSequence()
        j = numpy.ones(inDegSeq.shape[0])
        nonZeroInds = inDegSeq!=0
        zeroInds = inDegSeq==0

        if (inDegSeq[nonZeroInds] != j[nonZeroInds]).any():
            return False

        if numpy.sum(zeroInds) != 1:
            return False 

        #Find zero entry and ensure it has out-degree > 0
        root = numpy.nonzero(zeroInds)[0][0]
        neighbours = self.neighbours(root)

        if self.getNumVertices()>1 and neighbours.shape[0]==0:
            return False

        return True 

    def findTrees(self):
        """
        Returns a list of trees for a directed graph. The reason for only allowing
        directed graphs is that the root of a tree in an undirected graph is ambiguous.
        Each tree is represented by an list of indices of vertices in the graph.

        :returns: A list of trees (vertex indices) in the current graph sorted in descending order by size. 
        """
        if self.isUndirected():
            raise ValueError("Can only find trees on directed graphs")

        trees = []

        #Find all vertices with in-degree 0
        for i in range(0, self.getNumVertices()):
            if self.neighbourOf(i).shape[0] == 0:
                trees.append(self.depthFirstSearch(i))

        sortedIndices = numpy.array([len(x) for x in trees]).argsort()
        sortedTrees = []

        for i in reversed(list(range(len(trees)))):
            sortedTrees.append(list(trees[sortedIndices[i]]))

        return sortedTrees

    def toNetworkXGraph(self):
        """
        Convert this graph into a networkx Graph or DiGraph object, which requires
        networkx to be installed. Edge values are stored under the "value" index.
        Vertices are stored as indices with a "label" value being the corresponding
        vertex value. The type of vertex list is stored as a graph attribute under
        the index "VListType"

        :returns:  A networkx Graph or DiGraph object.
        """

        nxGraph = AbstractSingleGraph.toNetworkXGraph(self)
        nxGraph.graph["VListType"] = type(self.getVertexList())

        if nxGraph.graph["VListType"] == VertexList:
            nxGraph.graph["numFeatures"] = self.getVertexList().getNumFeatures()

        return nxGraph 

    def depthFirstSearch(self, root):
        """
        Depth first search starting from a particular vertex, based on the code found
        in Wikipedia. Returns the set of connected vertices.

        :param root: The index of the root vertex.
        :type root: :class:`int`

        :returns: A list of vertices connected to the input one via a path in the graph.
        """
        Parameter.checkIndex(root, 0, self.size)        
        
        toVisit = set()
        visited = set()

        toVisit.add(root)
        #adjacencyList, weights = self.adjacencyList()

        while len(toVisit) != 0:
            v = toVisit.pop()

            if v not in visited:
                visited.add(v)

            neighbours = self.neighbours(v)
            #neighbours = adjacencyList[v]
            toVisit = toVisit.union(set(neighbours).difference(visited))

        return list(visited)

    def getAllEdges(self):
        """
        Returns the set of edges of the current graph as a matrix in which each
        row corresponds to an edge. For an undirected graph, v1>=v2.

        :returns: A matrix with 2 columns, and each row corresponding to an edge.
        """
        edges = self.getAllDirEdges()

        if self.undirected and edges.shape[0] != 0:
            edges = edges[edges[:,0] >= edges[:,1], :]

        return edges

    def union(self, graph):
        """
        Take the union of the edges of this graph and the input graph. Resulting edge
        weights are ignored and only adjacencies are stored.

        :param graph: the input graph.
        :type graph: :class:`apgl.graph.SparseGraph`

        :returns: A new graph with the union of edges of the current one.
        """
        newGraph = self.add(graph)
        newGraph.W = newGraph.nativeAdjacencyMatrix()
        return newGraph

    def intersect(self, graph):
        """
        Take the intersection of the edges of this graph and the input graph.
        Resulting edge weights are ignored and only adjacencies are stored.

        :param graph: the input graph.
        :type graph: :class:`apgl.graph.SparseGraph`

        :returns: A new graph with the intersection of edges of the current plus graph.
        """
        newGraph = self.multiply(graph)
        newGraph.W = newGraph.nativeAdjacencyMatrix()
        return newGraph 

    @classmethod
    def fromNetworkXGraph(cls, networkXGraph):
        """
        Take a networkx Graph or DiGraph object, and return a subclass of AbstractMatrixGraph. Notice
        that networkx must be installed to use this function. The networkXGraph graph
        dict must have an attribute VListType which is the type of the VertexList used to
        construct the SparseGraph. Furthermore, only node attributes index by "label"
        are stored in the VertexList, and edge values are currently ignored.

        :returns: A networkx Graph or DiGraph object.
        """
        try:
            import networkx
        except ImportError:
            raise ImportError("toNetworkXGraph() requires networkx")

        if type(networkXGraph) == networkx.classes.graph.Graph:
            undirected = True
        elif type(networkXGraph) == networkx.classes.digraph.DiGraph:
            undirected = False
        else:
            raise ValueError("Unsupported NetworkX graph type")

        numVertices = networkXGraph.number_of_nodes()

        if "VListType" in networkXGraph.graph and networkXGraph.graph["VListType"] == VertexList:
            vList = networkXGraph.graph["VListType"](numVertices, networkXGraph.graph["numFeatures"])
        else:
            vList = GeneralVertexList(numVertices)

        graph = cls(vList, undirected)

        #Map from networkx nodes to an index
        nodeDict = {}

        #Set the vertices - note that vertex names are ignored
        for i in range(len(networkXGraph.nodes())):
            if "label" in networkXGraph.node[networkXGraph.nodes()[i]]:
                graph.setVertex(i, networkXGraph.node[networkXGraph.nodes()[i]]["label"])
            else:
                graph.setVertex(i, None)
            nodeDict[networkXGraph.nodes()[i]] = i

        #Set edges
        for i in range(len(networkXGraph.edges())):
            vertexIndex1 = nodeDict[networkXGraph.edges()[i][0]]
            vertexIndex2 = nodeDict[networkXGraph.edges()[i][1]]
            graph.addEdge(vertexIndex1, vertexIndex2)

        return graph

    def incidenceMatrix(self):
        """
        Return the incidence matrix of this graph as a scipy sparse matrix. The incidence
        matrix X is of size numVertices x numEdges, and has a 1 in element Xij = -1
        of edge j leaves vertex i, and Xij = 1 if edge j enters vertex i. Notice that
        for an undirected graph XX^T is the laplacian matrix. 
        """

        allEdges = self.getAllEdges()
        X = scipy.sparse.lil_matrix((self.getNumVertices(), allEdges.shape[0]))

        outgoing = -1
        incoming = 1

        for i in range(allEdges.shape[0]):
            X[allEdges[i, 0], i] = outgoing
            X[allEdges[i, 1], i] = incoming

        return X 

    def __getitem__(self, vertexIndices):
        """
        This is called when using square bracket notation and returns the value
        of the specified edge, e.g. graph[i, j] returns the edge between i and j.

        :param vertexIndices: a tuple of vertex indices (i, j)
        :type vertexIndices: :class:`tuple`

        :returns: The value of the edge. 
        """
        vertexIndex1, vertexIndex2 = vertexIndices
        return self.W[vertexIndex1, vertexIndex2]

    def __setitem__(self, vertexIndices, value):
        """
        This is called when using square bracket notation and sets the value
        of the specified edge, e.g. graph[i, j] = 1.

        :param vertexIndices: a tuple of vertex indices (i, j)
        :type vertexIndices: :class:`tuple`

        :param value: the value of the edge
        :type value: :class:`float`
        """
        vertexIndex1, vertexIndex2 = vertexIndices
        self.addEdge(vertexIndex1, vertexIndex2, value)

    def __getstate__(self): 
        tempFile = tempfile.NamedTemporaryFile(delete=False)
        tempFile.close()

        self.save(tempFile.name)
        infile = open(tempFile.name + ".zip", "rb")
        fileStr = infile.read()        
        infile.close() 
        
        try: 
            outputStr = base64.encodebytes(fileStr) 
        except AttributeError: 
            outputStr= base64.encodestring(fileStr)
            
        os.remove(tempFile.name)
        os.remove(tempFile.name + ".zip")
        
        return outputStr 
        
    def __setstate__(self, pkle): 
        tempFile = tempfile.NamedTemporaryFile(delete=False)
        tempFile.close()
        
        try: 
            zipstr = base64.decodebytes(pkle)
        except AttributeError: 
            zipstr = base64.decodestring(pkle)

        outFile = open(tempFile.name + ".zip" , "wb")
        outFile.write(zipstr) 
        outFile.close() 

        newGraph = self.load(tempFile.name)
        self.W = newGraph.W 
        self.undirected = newGraph.undirected 
        self.vList = newGraph.vList
        
        os.remove(tempFile.name)
        os.remove(outFile.name)

    def toDictGraph(self): 
        """
        Convert to a DictGraph object. Currently ignores vertex labels.
        
        :return graph: A DictGraph object.
        """
        edges = self.getAllEdges() 
        values = self.getEdgeValues(edges)
        graph = DictGraph(self.undirected)
        graph.addEdges(edges, values)
        
        return graph 

    vList = None
    undirected = None
    _wFilename = "weightMatrix.mtx"
    _metaFilename = "metaDict.dat"
    _verticesFilename = "vertices"
    _matExt = ".mtx"
    _boolExt = ".dir"
    
    vlist = property(getVertexList, doc="The vertex list")
    size = property(getNumVertices, doc="The number of vertices in the graph")


