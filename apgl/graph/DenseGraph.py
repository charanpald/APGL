
import scipy.io
import numpy
from apgl.graph.AbstractMatrixGraph import AbstractMatrixGraph
from apgl.graph.AbstractVertexList import AbstractVertexList 
from apgl.util.Parameter import Parameter
from apgl.util.SparseUtils import SparseUtils

class DenseGraph(AbstractMatrixGraph):
    def __init__(self, vertexList, undirected=True, dtype=numpy.float):
        """
        Create a DenseGraph with a given AbstractVertexList, and specify whether directed.

        :param vList: the initial set of vertices as a AbstractVertexList object.
        :type vList: :class:`apgl.graph.AbstractVertexList`

        :param undirected: a boolean variable to indicate if the graph is undirected.
        :type undirected: :class:`boolean`

        :param dtype: the data type for the weight matrix, e.g numpy.int8.
        """
        Parameter.checkClass(vertexList, AbstractVertexList)
        Parameter.checkBoolean(undirected)

        self.vList = vertexList
        self.W = numpy.zeros((vertexList.getNumVertices(), vertexList.getNumVertices()), dtype)
        self.undirected = undirected

    def getNumEdges(self):
        """
        Returns the total number of edges in this graph.
        """
        if self.undirected:
            return (numpy.flatnonzero(self.W).shape[0] + numpy.flatnonzero(numpy.diag(self.W)).shape[0])/2
        else: 
            return numpy.flatnonzero(self.W).shape[0]

    def getNumDirEdges(self):
        """
        Returns the number of edges, taking this graph as a directed graph. 
        """
        return numpy.flatnonzero(self.W).shape[0]
    
    def getWeightMatrix(self):
        """
        Return the weight matrix as a numpy array. 
        """
        return self.W

    def neighbours(self, vertexIndex):
        """
        Return an array of the indices of the neighbours of the given vertex.
        
        :param vertexIndex: the index of a vertex.
        :type vertexIndex: :class:`int`
        """
        Parameter.checkIndex(vertexIndex, 0, self.vList.getNumVertices())
        nonZeroIndices =  numpy.nonzero(self.W[vertexIndex, :])
        neighbourIndices = nonZeroIndices[0]
        
        return neighbourIndices

    def neighbourOf(self, vertexIndex):
        """
        Return an array of the indices of vertices than have an edge going to the input
        vertex.

        :param vertexIndex: the index of a vertex.
        :type vertexIndex: :class:`int`
        """
        Parameter.checkIndex(vertexIndex, 0, self.vList.getNumVertices())
        nonZeroIndices =  numpy.nonzero(self.W[:, vertexIndex])
        neighbourIndices = nonZeroIndices[0]

        return neighbourIndices

    def complement(self):
        """
        Returns a graph with identical vertices (same reference) to the current one, but with the
        complement of the set of edges. Edges that do not exist have weight 1.
        """
        newGraph = DenseGraph(self.vList, self.undirected)
        newGraph.W = (self.W == 0).astype(self.W.dtype)
        return newGraph

    def outDegreeSequence(self):
        """
        Return a vector of the (out)degree for each vertex.
        """
        degrees = numpy.zeros(self.W.shape[0], dtype=numpy.int32)

        for i in range(0, self.W.shape[0]):
            degrees[i] = numpy.sum(self.W[i, :] != 0)

        return degrees

    def inDegreeSequence(self):
        """
        Return a vector of the (out)degree for each vertex.
        """
        degrees = numpy.zeros(self.W.shape[0], dtype=numpy.int32)

        for i in range(0, self.W.shape[0]):
            degrees[i] = numpy.sum(self.W[:, i] != 0)

        return degrees 

    def subgraph(self, vertexIndices):
        """
        Pass in a list or set of vertexIndices and returns the subgraph containing
        those vertices only, and edges between them.

        :param vertexIndices: the indices of the subgraph vertices.
        :type vertexIndices: :class:`list`
        """
        Parameter.checkList(vertexIndices, Parameter.checkIndex, (0, self.getNumVertices()))
        vertexIndices = numpy.unique(numpy.array(vertexIndices)).tolist()
        vList = self.vList.subList(vertexIndices)

        subGraph = DenseGraph(vList, self.undirected, self.W.dtype)
        subGraph.W = self.W[vertexIndices, :][:, vertexIndices]

        return subGraph

    def add(self, graph):
        """
        Add the edge weights of the input graph to the current one. Results in a
        union of the edges.

        :param graph: the input graph.
        :type graph: :class:`apgl.graph.DenseGraph`

        :returns: A new graph with same vertex list and addition of edge weights 
        """
        Parameter.checkClass(graph, DenseGraph)
        if graph.getNumVertices() != self.getNumVertices():
            raise ValueError("Can only add edges from graph with same number of vertices")

        newGraph = DenseGraph(self.vList, self.undirected)
        newGraph.W = self.W + graph.W
        return newGraph

    def copy(self):
        """
        Returns a copy of this object, which also has a copy of the VertexList.
        """
        graph = DenseGraph(self.vList.copy(), self.undirected, self.W.dtype)
        graph.W = self.W.copy()
        return graph

    def multiply(self, graph):
        """
        Multiply the edge weights of the input graph to the current one. Results in an
        intersection of the edges.

        :param graph: the input graph.
        :type graph: :class:`apgl.graph.DenseGraph`

        :returns: A new graph with edge weights which are multiples of the current and graph
        """
        Parameter.checkClass(graph, DenseGraph)
        if graph.getNumVertices() != self.getNumVertices():
            raise ValueError("Can only add edges from graph with same number of vertices")

        newGraph = DenseGraph(self.vList, self.undirected)
        newGraph.W = self.W * graph.W
        return newGraph

    def intersect(self, graph):
        """
        Take the intersection of the edges of this graph and the input graph.
        Resulting edge weights are ignored and only adjacencies are stored.

        :param graph: the input graph.
        :type graph: :class:`apgl.graph.DenseGraph`

        :returns: A new graph with the intersection of edges of the current plus graph
        """
        newGraph = self.multiply(graph)
        newGraph.W = (newGraph.W != 0).astype(newGraph.W.dtype)
        return newGraph 

    def union(self, graph):
        """
        Take the union of the edges of this graph and the input graph. Resulting edge
        weights are ignored and only adjacencies are stored.

        :param graph: the input graph.
        :type graph: :class:`apgl.graph.DenseGraph`

        :returns: A new graph with the union of edges of the current one. 
        """
        newGraph = self.add(graph)
        newGraph.W = (newGraph.W != 0).astype(newGraph.W.dtype)

        return newGraph

    def weightMatrixDType(self):
        """
        :returns: the dtype of the matrix used to store edge weights.
        """
        return self.W.dtype

    def setDiff(self, graph):
        """
        Find the edges in the current graph which are not present in the input
        graph. Replaces the edges in the current graph with adjacencies.

        :param graph: the input graph.
        :type graph: :class:`apgl.graph.DenseGraph`

        :returns: The graph which is the set difference of the edges of this graph and graph.
        """
        Parameter.checkClass(graph, DenseGraph)
        if graph.getNumVertices() != self.getNumVertices():
            raise ValueError("Can only add edges from graph with same number of vertices")
        if self.undirected != graph.undirected:
            raise ValueError("Both graphs must be either undirected or directed")

        A1 = self.adjacencyMatrix()
        A2 = graph.adjacencyMatrix()
        A1 = A1 - A2
        A1 = (A1 + numpy.abs(A1**2))/2
        
        newGraph = DenseGraph(self.vList, self.undirected)
        newGraph.W = A1
        return newGraph

    def getAllDirEdges(self):
        """
        Returns the set of directed edges of the current graph as a matrix in which each
        row corresponds to an edge. For an undirected graph, there is an edge from
        v1 to v2 and from v2 to v1 if v2!=v1.

        :returns: A matrix with 2 columns, and each row corresponding to an edge.
        """
        (rows, cols) = numpy.nonzero(self.W)
        edges = numpy.c_[rows, cols]

        return edges

    @staticmethod
    def loadMatrix(filename):
        M = scipy.io.mmread(filename)
        if scipy.sparse.issparse(M):
            M = M.todense()
        return M 

    def saveMatrix(self, W, filename):
        scipy.io.mmwrite(filename, W)

    def setWeightMatrix(self, W):
        """
        Set the weight matrix of this graph. Requires as input an ndarray or
        a scipy sparse matrix with the same dimensions as the current weight
        matrix. Edges are represented by non-zero edges.

        :param W: The weight matrix to use.
        :type W: :class:`ndarray` or :class:`scipy.sparse` matrix
        """
        if W.shape != (self.vList.getNumVertices(), self.vList.getNumVertices()):
            raise ValueError("Weight matrix has wrong shape : " + str(W.shape))

        if self.undirected and type(W) == numpy.ndarray and (W != W.T).any():
            raise ValueError("Weight matrix of undirected graph must be symmetric")

        if self.undirected and scipy.sparse.issparse(W) and not SparseUtils.equals(W, W.T):
            raise ValueError("Weight matrix of undirected graph must be symmetric")

        if scipy.sparse.issparse(W):
            W = W.todense()

        self.W = numpy.array(W)

    undirected = None
    vList = None
    W = None
    