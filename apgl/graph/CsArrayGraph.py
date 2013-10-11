
import scipy.io
import numpy
import sppy 
from apgl.graph.AbstractMatrixGraph import AbstractMatrixGraph
from apgl.graph.AbstractVertexList import AbstractVertexList
from apgl.graph import GeneralVertexList  
from apgl.util.Parameter import Parameter
from apgl.util.SparseUtils import SparseUtils

class CsArrayGraph(AbstractMatrixGraph):
    def __init__(self, vertices, undirected=True, dtype=numpy.float):
        """
        Create a sparse graph using sppy csarray with a given AbstractVertexList, and specify whether directed.

        :param vertices: the initial set of vertices as a AbstractVertexList object, or an int to specify the number of vertices in which case vertices are stored in a GeneralVertexList.  
        
        :param undirected: a boolean variable to indicate if the graph is undirected.
        :type undirected: :class:`boolean`

        :param dtype: the data type for the weight matrix, e.g numpy.int8.
        """
        Parameter.checkBoolean(undirected)

        if isinstance(vertices, AbstractVertexList):
            self.vList = vertices
        elif isinstance(vertices, int): 
            self.vList = GeneralVertexList(vertices)
        else: 
            raise ValueError("Invalid vList parameter: " + str(vertices))

        self.W = sppy.csarray((self.vList.getNumVertices(), self.vList.getNumVertices()), dtype)
        self.undirected = undirected

    def getNumEdges(self):
        """
        Returns the total number of edges in this graph.
        """
        if self.undirected:
            return (self.W.getnnz() + numpy.flatnonzero(self.W.diag()).shape[0])/2
        else: 
            return self.W.getnnz()

    def getNumDirEdges(self):
        """
        Returns the number of edges, taking this graph as a directed graph. 
        """
        return self.W.getnnz()
    
    def getWeightMatrix(self):
        """
        Return the weight matrix as a numpy array. 
        """
        return self.W.toarray()

    def neighbours(self, vertexIndex):
        """
        Return an array of the indices of the neighbours of the given vertex.
        
        :param vertexIndex: the index of a vertex.
        :type vertexIndex: :class:`int`
        """
        Parameter.checkIndex(vertexIndex, 0, self.vList.getNumVertices())
        nonZeroIndices =  self.W[vertexIndex, :].nonzero()
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
        newGraph = CsArrayGraph(self.vList, self.undirected)
        newGraph.W = sppy.ones((newGraph.W.shape)) 
        newGraph.W[self.W.nonzero()] = 0
        newGraph.W.prune()
        newGraph.W.compress()
        return newGraph

    def outDegreeSequence(self):
        """
        Return a vector of the (out)degree for each vertex.
        """
        degrees = numpy.zeros(self.W.shape[0], dtype=numpy.int32)

        for i in range(0, self.W.shape[0]):
            degrees[i] = self.W[i, :].getnnz()

        return degrees

    def inDegreeSequence(self):
        """
        Return a vector of the (out)degree for each vertex.
        """
        degrees = numpy.zeros(self.W.shape[0], dtype=numpy.int32)

        for i in range(0, self.W.shape[0]):
            degrees[i] = self.W[:, i].getnnz()

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

        subGraph = CsArrayGraph(vList, self.undirected, self.W.dtype)
        subGraph.W = self.W[vertexIndices, :][:, vertexIndices]

        return subGraph

    def add(self, graph):
        """
        Add the edge weights of the input graph to the current one. Results in a
        union of the edges.

        :param graph: the input graph.
        :type graph: :class:`apgl.graph.CsArrayGraph`

        :returns: A new graph with same vertex list and addition of edge weights 
        """
        Parameter.checkClass(graph, CsArrayGraph)
        if graph.getNumVertices() != self.getNumVertices():
            raise ValueError("Can only add edges from graph with same number of vertices")

        newGraph = CsArrayGraph(self.vList, self.undirected)
        newGraph.W = self.W + graph.W
        return newGraph

    def copy(self):
        """
        Returns a copy of this object, which also has a copy of the VertexList.
        """
        graph = CsArrayGraph(self.vList.copy(), self.undirected, self.W.dtype)
        graph.W = self.W.copy()
        return graph

    def multiply(self, graph):
        """
        Multiply the edge weights of the input graph to the current one. Results in an
        intersection of the edges.

        :param graph: the input graph.
        :type graph: :class:`apgl.graph.CsArrayGraph`

        :returns: A new graph with edge weights which are multiples of the current and graph
        """
        Parameter.checkClass(graph, CsArrayGraph)
        if graph.getNumVertices() != self.getNumVertices():
            raise ValueError("Can only add edges from graph with same number of vertices")

        newGraph = CsArrayGraph(self.vList, self.undirected)
        newGraph.W = self.W.hadamard(graph.W)
        return newGraph

    def intersect(self, graph):
        """
        Take the intersection of the edges of this graph and the input graph.
        Resulting edge weights are ignored and only adjacencies are stored.

        :param graph: the input graph.
        :type graph: :class:`apgl.graph.CsArrayGraph`

        :returns: A new graph with the intersection of edges of the current plus graph
        """
        newGraph = self.multiply(graph)
        newGraph.W[newGraph.W.nonzero()] = 1
        return newGraph 

    def union(self, graph):
        """
        Take the union of the edges of this graph and the input graph. Resulting edge
        weights are ignored and only adjacencies are stored.

        :param graph: the input graph.
        :type graph: :class:`apgl.graph.CsArrayGraph`

        :returns: A new graph with the union of edges of the current one. 
        """
        newGraph = self.add(graph)
        newGraph.W[newGraph.W.nonzero()] = 1

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
        :type graph: :class:`apgl.graph.CsArrayGraph`

        :returns: The graph which is the set difference of the edges of this graph and graph.
        """
        Parameter.checkClass(graph, CsArrayGraph)
        if graph.getNumVertices() != self.getNumVertices():
            raise ValueError("Can only add edges from graph with same number of vertices")
        if self.undirected != graph.undirected:
            raise ValueError("Both graphs must be either undirected or directed")

        A1 = self.adjacencyMatrix()
        A2 = graph.adjacencyMatrix()
        A1 = A1 - A2
        A1 = (A1 + numpy.abs(A1**2))/2
        
        newGraph = CsArrayGraph(self.vList, self.undirected)
        newGraph.W = sppy.csarray(A1)
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
        if type(M) == numpy.ndarray:
            M2 = sppy.csarray(M)
        elif scipy.sparse.issparse(M):
            M2 = sppy.csarray(M.shape, dtype=M.dtype)
            M2[M.nonzero()] = M.data 
                
        return M2 

    def saveMatrix(self, W, filename):
        W = W.toScipyCsc()
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

    def removeAllEdges(self):
        """
        Removes all edges from this graph. 
        """
        self.W.setZero()

    def setWeightMatrixSparse(self, W):
        """
        Set the weight matrix of this graph. Requires as input a scipy sparse matrix with the
        same dimensions as the current weight matrix. Edges are represented by
        non-zero edges.

        :param W:  The weight matrix to use. 
        """      
        self.W[W.nonzero()] = W.data

    undirected = None
    vList = None
    W = None
    