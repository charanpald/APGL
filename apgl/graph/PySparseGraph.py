import numpy
import scipy.sparse as sparse
from apgl.graph.AbstractMatrixGraph import AbstractMatrixGraph
from apgl.graph.AbstractVertexList import AbstractVertexList
from apgl.util.Parameter import Parameter
from apgl.util.PySparseUtils import PySparseUtils
from pysparse import spmatrix
from pysparse.sparse.pysparseMatrix import PysparseMatrix


class PySparseGraph(AbstractMatrixGraph):
    '''
    Represents a graph, which can be directed or undirected, and has weights
    on the edges. Memory usage is efficient for sparse graphs. The list of vertices
    is immutable (see VertexList), however edges can be added or removed. Only
    non-zero edges can be added.  Uses Pysparse as the underlying matrix
    representation.
    '''
    def __init__(self, vList, undirected=True, sizeHint=1000):
        """
        Create a PySparseGraph with a given AbstractVertexList, and specify whether
        it is directed. 

        :param vList: the initial set of vertices as a AbstractVertexList object.
        :type vList: :class:`apgl.graph.AbstractVertexList`

        :param undirected: a boolean variable to indicate if the graph is undirected.
        :type undirected: :class:`boolean`

        :param sizeHint: the expected number of edges in the graph for efficient memory usage.
        :type sizeHint: :class:`int`
        """
        Parameter.checkClass(vList, AbstractVertexList)
        Parameter.checkBoolean(undirected)

        self.vList = vList
        self.undirected = undirected

        #Should use ll_mat_sym for undirected graphs but it has several unimplemented methods 
        self.W = spmatrix.ll_mat(vList.getNumVertices(), vList.getNumVertices(), sizeHint)

    def neighbours(self, vertexIndex):
        """
        Return an array of the indices of neighbours. In the case of a directed
        graph it is an array of those vertices connected by an edge from the current
        one.

        :param vertexIndex: the index of a vertex.
        :type vertexIndex: :class:`int`

        :returns: An array of the indices of all neigbours of the input vertex.
        """
        Parameter.checkIndex(vertexIndex, 0, self.vList.getNumVertices())
        neighbours = PySparseUtils.nonzero(self.W[int(vertexIndex), :])[1]

        return numpy.array(neighbours)

    def neighbourOf(self, vertexIndex):
        """
        Return an array of the indices of vertices than have an edge going to the input
        vertex.

        :param vertexIndex: the index of a vertex.
        :type vertexIndex: :class:`int`

        :returns: An array of the indices of all vertices with an edge towards the input vertex.
        """
        Parameter.checkIndex(vertexIndex, 0, self.vList.getNumVertices())
        neighbours = PySparseUtils.nonzero(self.W[:, vertexIndex])[0]

        return numpy.array(neighbours)

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
        vertexIndex1 = int(vertexIndex1)
        vertexIndex2 = int(vertexIndex2)

        if edge == 0 or edge == float('inf'):
            raise ValueError("Cannot add a zero or infinite edge")

        if self.undirected:
            self.W[vertexIndex1, vertexIndex2] = edge
            self.W[vertexIndex2, vertexIndex1] = edge
        else:
            self.W[vertexIndex1, vertexIndex2] = edge

    def getNumEdges(self):
        """
        Returns the total number of edges in this graph.
        """
        if self.getNumVertices()==0:
            return 0

        if self.undirected:
            inds = numpy.arange(self.getNumVertices())
            #d = spmatrix.ll_mat(self.getNumVertices(), 1)
            d = numpy.zeros(self.getNumVertices())
            self.W.take(d, inds, inds)

            return (self.W.nnz + numpy.sum(d!=0))/2
        else:
            return self.W.nnz

    def getNumDirEdges(self):
        """
        Returns the number of edges, taking this graph as a directed graph.
        """
        return self.W.nnz

    def outDegreeSequence(self):
        """
        Return a vector of the (out)degree for each vertex.
        """
        A = self.nativeAdjacencyMatrix()
        j = spmatrix.ll_mat(self.vList.getNumVertices(), 1)
        j[:, 0] = 1 

        degrees = spmatrix.matrixmultiply(A, j)
        degrees = PysparseMatrix(matrix=degrees)
        degrees = numpy.array(degrees.getNumpyArray().ravel(), numpy.int)
        return degrees

    def inDegreeSequence(self):
        """
        Return a vector of the (in)degree sequence for each vertex.
        """
        A = self.nativeAdjacencyMatrix()
        j = spmatrix.ll_mat(self.vList.getNumVertices(), 1)
        j[:, 0] = 1

        degrees = spmatrix.dot(A, j)
        degrees = PysparseMatrix(matrix=degrees)
        degrees = numpy.array(degrees.getNumpyArray().ravel(), numpy.int)
        return degrees

    def nativeAdjacencyMatrix(self):
        """
        Return the adjacency matrix in sparse format.
        """
        A = spmatrix.ll_mat(self.vList.getNumVertices(), self.vList.getNumVertices())

        nonzeros = PySparseUtils.nonzero(self.W)
        A.put(1, nonzeros[0], nonzeros[1])
        return A

    def subgraph(self, vertexIndices):
        """
        Pass in a list or set of vertexIndices and returns the subgraph containing
        those vertices only, and edges between them.

        :param vertexIndices: the indices of the subgraph vertices.
        :type vertexIndices: :class:`list`

        :returns: A new PySparseGraph containing only vertices and edges from vertexIndices
        """
        Parameter.checkList(vertexIndices, Parameter.checkIndex, (0, self.getNumVertices()))
        vertexIndices = numpy.unique(numpy.array(vertexIndices))
        vList = self.vList.subList(vertexIndices.tolist())

        subGraph = PySparseGraph(vList, self.undirected)
        
        if len(vertexIndices) != 0:
            subGraph.W = self.W[vertexIndices, vertexIndices]

        return subGraph

    def getWeightMatrix(self):
        """
        Return the weight matrix in dense format. Warning: should not be used
        unless sufficient memory is available to store the dense matrix.

        :returns: A numpy.ndarray weight matrix.
        """
        W = PysparseMatrix(matrix=self.W)
        return W.getNumpyArray()

    def getAllDirEdges(self):
        """
        Returns the set of directed edges of the current graph as a matrix in which each
        row corresponds to an edge. For an undirected graph, there is an edge from
        v1 to v2 and from v2 to v1 if v2!=v1.

        :returns: A matrix with 2 columns, and each row corresponding to an edge.
        """
        (rows, cols) = PySparseUtils.nonzero(self.W)
        edges = numpy.c_[rows, cols]

        return edges


    def add(self, graph):
        """
        Add the edge weights of the input graph to the current one. Results in a
        union of the edges.

        :param graph: the input graph.
        :type graph: :class:`apgl.graph.PySparseGraph`

        :returns: A new graph with same vertex list and addition of edge weights
        """
        Parameter.checkClass(graph, PySparseGraph)
        if graph.getNumVertices() != self.getNumVertices():
            raise ValueError("Can only add edges from graph with same number of vertices")
        if self.undirected != graph.undirected:
            raise ValueError("Both graphs must be either undirected or directed")

        newGraph = PySparseGraph(self.vList, self.undirected)
        newGraph.W = self.W.copy()
        newGraph.W.shift(1, graph.W)

        return newGraph

    def copy(self):
        """
        Returns a copy of this object, which also has a copy of the VertexList.
        """
        newGraph = PySparseGraph(self.vList.copy(), self.undirected)
        newGraph.W = self.W.copy()
        return newGraph

    def removeAllEdges(self):
        """
        Removes all edges from this graph.
        """
        #Not sure why this doesn't work 
        #self.W.scale(0)
        self.W = spmatrix.ll_mat(self.getNumVertices(), self.getNumVertices())

    def setWeightMatrix(self, W):
        """
        Set the weight matrix of this graph. Requires as input an ndarray with the
        same dimensions as the current weight matrix. Edges are represented by
        non-zero edges.

        :param W: The name of the file to load.
        :type W: :class:`ndarray`
        """
        #Parameter.checkClass(W, numpy.ndarray)

        if W.shape != (self.vList.getNumVertices(), self.vList.getNumVertices()):
            raise ValueError("Weight matrix has wrong shape : " + str(W.shape))

        if self.undirected and type(W) == numpy.ndarray and (W != W.T).any():
            raise ValueError("Weight matrix of undirected graph must be symmetric")

        self.W = spmatrix.ll_mat(self.getNumVertices(), self.getNumVertices())
        
        if type(W) == numpy.ndarray:         
            rowInds, colInds = numpy.nonzero(W)
            self.W.put(W[(rowInds, colInds)], rowInds, colInds)
        elif sparse.issparse(W): 
            self.setWeightMatrixSparse(W)
        else: 
            raise ValueError("Invalid matrix type: " + str(type(W)))

    def weightMatrixType(self):
        """
        Returns the type of the sparse matrix used to store edge weights.
        """
        return type(self.W)

    def complement(self):
        """
        Returns a graph with identical vertices (same reference) to the current
        one, but with the complement of the set of edges. Edges that do not exist
        have weight 1. This makes a sparse graph dense.

        :returns: A new graph with edges complmenting the current one.
        """

        newGraph = PySparseGraph(self.vList, self.undirected)
        newGraph.W[:, :] = 1

        A = self.nativeAdjacencyMatrix()
        newGraph.W.shift(-1, A)

        return newGraph

    def multiply(self, graph):
        """
        Multiply the edge weights of the input graph to the current one. Results in an
        intersection of the edges.

        :param graph: the input graph.
        :type graph: :class:`apgl.graph.PySparseGraph`

        :returns: A new graph with edge weights which are multiples of the current and graph
        """
        Parameter.checkClass(graph, PySparseGraph)
        if graph.getNumVertices() != self.getNumVertices():
            raise ValueError("Can only add edges from graph with same number of vertices")
        if self.undirected != graph.undirected:
            raise ValueError("Both graphs must be either undirected or directed")

        if self.W.nnz < graph.W.nnz:
            (rows, cols) = PySparseUtils.nonzero(self.W)
        else:
            (rows, cols) = PySparseUtils.nonzero(graph.W)

        arr1 = numpy.zeros(len(rows))
        arr2 = numpy.zeros(len(rows))
        self.W.take(arr1, rows, cols)
        graph.W.take(arr2, rows, cols)

        arr1 = arr1 * arr2

        newGraph = PySparseGraph(self.vList, self.undirected)
        newGraph.W.put(arr1, rows, cols)
        return newGraph

    def setDiff(self, graph):
        """
        Find the edges in the current graph which are not present in the input
        graph.

        :param graph: the input graph.
        :type graph: :class:`apgl.graph.PySparseGraph`

        :returns: A new graph with edges from the current graph and not in the input graph.
        """
        Parameter.checkClass(graph, PySparseGraph)
        if graph.getNumVertices() != self.getNumVertices():
            raise ValueError("Can only add edges from graph with same number of vertices")
        if self.undirected != graph.undirected:
            raise ValueError("Both graphs must be either undirected or directed")

        A1 = self.nativeAdjacencyMatrix()
        A2 = graph.nativeAdjacencyMatrix()

        (rows, cols) = PySparseUtils.nonzero(A1)
        arr1 = numpy.zeros(len(rows))
        arr2 = numpy.zeros(len(rows))

        A1.take(arr1, rows, cols)
        A2.take(arr2, rows, cols)
        arr1 = arr1 - arr2

        A1.put(arr1, rows, cols)

        newGraph = PySparseGraph(self.vList, self.undirected)
        newGraph.W = A1
        return newGraph

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
        vertexIndex1 = int(vertexIndex1)
        vertexIndex2 = int(vertexIndex2)

        if self.W[vertexIndex1, vertexIndex2]==0:
            return None
        else:
            return self.W[vertexIndex1, vertexIndex2]

    @staticmethod
    def loadMatrix(filename):
        return spmatrix.ll_mat_from_mtx(filename)

    def saveMatrix(self, W, filename):
        W.export_mtx(filename)
        
    def setWeightMatrixSparse(self, W):
        """
        Set the weight matrix of this graph. Requires as input a scipy sparse matrix with the
        same dimensions as the current weight matrix. Edges are represented by
        non-zero edges.

        :param W:  The weight matrix to use. 
        """
        if not sparse.issparse(W):
            raise ValueError("Input must be a sparse matrix, not " + str(type(W)))

        if W.shape != (self.vList.getNumVertices(), self.vList.getNumVertices()):
            raise ValueError("Weight matrix has wrong shape : " + str(W.shape))

        if self.undirected and (W - W.transpose()).nonzero()[0].shape[0]:
            raise ValueError("Weight matrix of undirected graph must be symmetric")
        
        self.W = spmatrix.ll_mat(W.shape[0], W.shape[0], W.getnnz())
        rowInds, colInds = W.nonzero()
        
        for i in range(rowInds.shape[0]):
            self.W[int(rowInds[i]), int(colInds[i])] = W[rowInds[i], colInds[i]]
        
        