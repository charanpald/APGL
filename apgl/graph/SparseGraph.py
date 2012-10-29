

from apgl.graph.AbstractMatrixGraph import AbstractMatrixGraph
from apgl.graph.AbstractVertexList import AbstractVertexList
from apgl.graph.GeneralVertexList import GeneralVertexList
from apgl.graph.VertexList import VertexList 
from apgl.util.Util import Util
from apgl.util.SparseUtils import SparseUtils
from apgl.util.Parameter import Parameter
import scipy.sparse as sparse
import scipy.io
import numpy

class SparseGraph(AbstractMatrixGraph):
    '''
    Represents a graph, which can be directed or undirected, and has weights
    on the edges. Memory usage is efficient for sparse graphs. The list of vertices
    is immutable (see VertexList), however edges can be added or removed. Only
    non-zero edges can be added. Uses scipy.sparse for the underlying matrix
    representation. 
    '''
    def __init__(self, vList, undirected=True, W=None):
        """
        Create a SparseGraph with a given AbstractVertexList, and specify whether 
        it is directed. One can optionally pass in an empty sparse matrix W which
        is used as the weight matrix of the graph. Different kinds of sparse matrix
        can impact the speed of various operations. The currently supported sparse
        matrix types are: lil_matrix, csr_matrix, csc_matrix and dok_matrix. 

        :param vList: the initial set of vertices as a AbstractVertexList object.
        :type vList: :class:`apgl.graph.AbstractVertexList`
        
        :param undirected: a boolean variable to indicate if the graph is undirected.
        :type undirected: :class:`boolean`

        :param W: an empty sparse matrix of the same size as vList, or None to create the default one.
        """
        Parameter.checkClass(vList, AbstractVertexList)
        Parameter.checkBoolean(undirected)
        if W != None and not (sparse.issparse(W) and W.getnnz()==0 and W.shape == (vList.getNumVertices(), vList.getNumVertices())):
            raise ValueError("Input argument W must be None or empty sparse matrix of size " + str(vList.getNumVertices()) )

        self.vList = vList
        self.undirected = undirected

        #Terrible hack alert:  can't create a zero size sparse matrix, so we settle
        #for one of size 1. Better is to create a new class. 
        if vList.getNumVertices() == 0 and W == None:
            self.W = sparse.csr_matrix((1, 1))
        elif W == None:
            self.W = sparse.csr_matrix((vList.getNumVertices(), vList.getNumVertices()))
        else:
            self.W = W 
        
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
        #neighbours = self.W[vertexIndex, :].nonzero()[1]
        neighbours = self.W.getrow(vertexIndex).nonzero()[1]
        #neighbours = numpy.nonzero(self.W.getrow(vertexIndex).toarray())[1]

        return neighbours

    def neighbourOf(self, vertexIndex):
        """
        Return an array of the indices of vertices than have an edge going to the input
        vertex.

        :param vertexIndex: the index of a vertex.
        :type vertexIndex: :class:`int`

        :returns: An array of the indices of all vertices with an edge towards the input vertex.
        """
        Parameter.checkIndex(vertexIndex, 0, self.vList.getNumVertices())
        nonZeroInds = self.W[:, vertexIndex].nonzero()
        neighbours = nonZeroInds[0]

        return neighbours
    
    def getNumEdges(self):
        """
        :returns: the total number of edges in this graph.
        """
        if self.getNumVertices()==0:
            return 0 

        #Note that self.W.getnnz() doesn't seem to work correctly 
        if self.undirected == True:
            return (self.W.nonzero()[0].shape[0] + numpy.sum(SparseUtils.diag(self.W) != 0))/2
        else: 
            return self.W.nonzero()[0].shape[0]

    def getNumDirEdges(self):
        """
        :returns: the number of edges, taking this graph as a directed graph.
        """
        return self.W.nonzero()[0].shape[0]
    
    def outDegreeSequence(self):
        """
        :returns: a vector of the (out)degree sequence for each vertex.
        """
        A = self.nativeAdjacencyMatrix()
        degrees = numpy.array(A.sum(1), dtype=numpy.int32).ravel()

        return degrees 

    def inDegreeSequence(self):
        """
        :returns: a vector of the (in)degree sequence for each vertex.
        """
        A = self.nativeAdjacencyMatrix()
        degrees = numpy.array(A.sum(0), dtype=numpy.int32).ravel()

        return degrees 
    
    def subgraph(self, vertexIndices):
        """
        Pass in a list or set of vertexIndices and returns the subgraph containing
        those vertices only, and edges between them. The subgraph indices correspond
        to the sorted input indices. 

        :param vertexIndices: the indices of the subgraph vertices.
        :type vertexIndices: :class:`list`

        :returns: A new SparseGraph containing only vertices and edges from vertexIndices
        """
        Parameter.checkList(vertexIndices, Parameter.checkIndex, (0, self.getNumVertices()))
        vertexIndices = numpy.unique(numpy.array(vertexIndices)).tolist()
        vList = self.vList.subList(vertexIndices)

        subGraph = SparseGraph(vList, self.undirected)
        
        if len(vertexIndices) != 0:
            subGraph.W = self.W[vertexIndices, :][:, vertexIndices]

        return subGraph

    def getWeightMatrix(self):
        """
        Return the weight matrix in dense format. Warning: should not be used
        unless sufficient memory is available to store the dense matrix.

        :returns: A numpy.ndarray weight matrix.
        """
        if self.getVertexList().getNumVertices() != 0: 
            return self.W.toarray()
        else: 
            return numpy.zeros((0, 0))

    def getSparseWeightMatrix(self):
        """
        Returns the original sparse weight matrix.

        :returns: A scipy.sparse weight matrix.
        """

        return self.W

    def add(self, graph):
        """
        Add the edge weights of the input graph to the current one. Results in a
        union of the edges.

        :param graph: the input graph.
        :type graph: :class:`apgl.graph.SparseGraph`

        :returns: A new graph with same vertex list and addition of edge weights 
        """
        Parameter.checkClass(graph, SparseGraph)
        if graph.getNumVertices() != self.getNumVertices():
            raise ValueError("Can only add edges from graph with same number of vertices")
        if self.undirected != graph.undirected:
            raise ValueError("Both graphs must be either undirected or directed")

        #The ideal way is to add both weight matrices together, but this results in a csr
        #We'll just do this manually
        nonZeros = numpy.nonzero(graph.W)
        newGraph = SparseGraph(self.vList, self.undirected)
        newGraph.W = self.W.copy()

        for i in range(len(nonZeros[0])):
            ind1 = nonZeros[0][i]
            ind2 = nonZeros[1][i]
            newGraph.W[ind1, ind2] = self.W[ind1, ind2] +  graph.W[ind1, ind2]

        return newGraph

    def multiply(self, graph):
        """
        Multiply the edge weights of the input graph to the current one. Results in an
        intersection of the edges.

        :param graph: the input graph.
        :type graph: :class:`apgl.graph.SparseGraph`

        :returns: A new graph with edge weights which are multiples of the current and graph
        """
        Parameter.checkClass(graph, SparseGraph)
        if graph.getNumVertices() != self.getNumVertices():
            raise ValueError("Can only add edges from graph with same number of vertices")
        if self.undirected != graph.undirected:
            raise ValueError("Both graphs must be either undirected or directed")

        newGraph = SparseGraph(self.vList, self.undirected)
        newGraph.W = self.W.multiply(graph.W)
        return newGraph

    def copy(self):
        """
        Returns a copy of this object, which also has a copy of the AbstractVertexList.
        """
        newGraph = SparseGraph(self.vList.copy(), self.undirected)
        newGraph.W = self.W.copy()
        return newGraph

    def complement(self):
        """
        Returns a graph with identical vertices (same reference) to the current
        one, but with the complement of the set of edges. Edges that do not exist
        have weight 1. This makes a sparse graph dense.

        :returns: A new graph with edges complmenting the current one. 
        """
        newGraph = SparseGraph(self.vList, self.undirected)
        newGraph.W = self.weightMatrixType()(numpy.ones((self.vList.getNumVertices(), self.vList.getNumVertices())))

        A = self.nativeAdjacencyMatrix()
        newGraph.W = newGraph.W - A

        return newGraph

    def setWeightMatrix(self, W):
        """
        Set the weight matrix of this graph. Requires as input an ndarray or 
        a scipy sparse matrix with the same dimensions as the current weight
        matrix. Edges are represented by non-zero edges.

        :param W: The weight matrix to use. 
        :type W: :class:`ndarray` or :class:`scipy.sparse` matrix
        """
        #Parameter.checkClass(W, numpy.ndarray)

        if W.shape != (self.vList.getNumVertices(), self.vList.getNumVertices()):
            raise ValueError("Weight matrix has wrong shape : " + str(W.shape))

        if self.undirected and type(W) == numpy.ndarray and (W != W.T).any():
            raise ValueError("Weight matrix of undirected graph must be symmetric")

        if self.undirected and scipy.sparse.issparse(W) and not SparseUtils.equals(W, W.T):
            raise ValueError("Weight matrix of undirected graph must be symmetric")

        self.W = self.weightMatrixType()(W)

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

        self.W = W

    def weightMatrixType(self):
        """
        :returns: the type of the sparse matrix used to store edge weights.
        """
        return type(self.W)

    def removeEdge(self, vertexIndex1, vertexIndex2):
        """
        Remove an edge between two vertices.

        :param vertexIndex1: The index of the first vertex.
        :type vertexIndex1: :class:`int`

        :param vertexIndex2: The index of the second vertex.
        :type vertexIndex2: :class:`int`
        """
        super(SparseGraph, self).removeEdge(vertexIndex1, vertexIndex2)

        self.W.eliminate_zeros()

    def nativeAdjacencyMatrix(self):
        """
        :returns: the adjacency matrix in the native sparse format.
        """
        A = self.W/self.W
        return A

    def setDiff(self, graph):
        """
        Find the edges in the current graph which are not present in the input
        graph. 

        :param graph: the input graph.
        :type graph: :class:`apgl.graph.SparseGraph`

        :returns: A new graph with edges from the current graph and not in the input graph. 
        """
        Parameter.checkClass(graph, SparseGraph)
        if graph.getNumVertices() != self.getNumVertices():
            raise ValueError("Can only add edges from graph with same number of vertices")
        if self.undirected != graph.undirected:
            raise ValueError("Both graphs must be either undirected or directed")

        A1 = self.nativeAdjacencyMatrix()
        A2 = graph.nativeAdjacencyMatrix()
        A1 = A1 - A2

        A = (A1 + A1.multiply(A1))/2
        A.prune()

        newGraph = SparseGraph(self.vList, self.undirected)
        newGraph.W = A
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
        W = scipy.io.mmread(filename)
        return W.tolil()

    def saveMatrix(self, W, filename):
        scipy.io.mmwrite(filename, W)

    def removeAllEdges(self):
        """
        Removes all edges from this graph.
        """
        self.W = self.W*0

        #Weirdly we get nan values for the edges after doing the above line 
        if sparse.isspmatrix_csr(self.W) or sparse.isspmatrix_csc(self.W):
            self.W.eliminate_zeros()

    def concat(self, graph):
        """
        Take a new graph and concatenate it to the current one. Returns a new graph
        of the concatenated graphs with this graphs vertices first in the new list of
        vertices.

        :param graph: the input graph.
        :type graph: :class:`apgl.graph.SparseGraph`
        """
        Parameter.checkClass(graph, SparseGraph)
        if type(graph.getVertexList()) != type(self.getVertexList()):
            raise ValueError("Vertex lists must be of same type")
        if graph.isUndirected() != self.isUndirected():
            raise ValueError("Graphs must be of the same directed type")

        numVertices = self.getNumVertices() + graph.getNumVertices()
        vList = GeneralVertexList(numVertices)
        vList.setVertices(self.getVertexList().getVertices(), list(range(self.getNumVertices())))
        vList.setVertices(graph.getVertexList().getVertices(), list(range(self.getNumVertices(), numVertices)))
        newGraph = SparseGraph(vList)
        
        W = scipy.sparse.bmat([[self.W, None], [None, graph.W]], format="csr")
        newGraph.setWeightMatrixSparse(W)

        return newGraph 

    def normalisedLaplacianSym(self, outDegree=True, sparse=False):
        """
        Compute the normalised symmetric laplacian matrix using L = I - D^-1/2 W D^-1/2,
        in which W is the weight matrix and D_ii is the sum of the ith vertices weights.

        :param outDegree: whether to use the out-degree for the computation of the degree matrix
        :type outDegree: :class:`bool`

        :param sparse: whether to return a sparse matrix or numpy array
        :type sparse: :class:`bool`

        :returns:  A normalised symmetric laplacian matrix
        """
        W = self.getSparseWeightMatrix()

        if outDegree:
            degrees = numpy.array(W.sum(1)).ravel()
        else:
            degrees = numpy.array(W.sum(1)).ravel()

        L = self.weightMatrixType()((self.getNumVertices(), self.getNumVertices()))
        L.setdiag(numpy.ones(self.getNumVertices()))

        D2 = self.weightMatrixType()((self.getNumVertices(), self.getNumVertices()))
        D2.setdiag((degrees + (degrees==0))**-0.5)

        L = L - D2.dot(W).dot(D2)

        if sparse == True:
            return L
        else:
            return L.toarray()

    def laplacianMatrix(self, outDegree=True, sparse=False):
        """
        Return the Laplacian matrix of this graph, which is defined as L_{ii} = deg(i)
        L_{ij} = -1 if an edge between i and j, otherwise L_{ij} = 0 . For a directed
        graph one can specify whether to use the out-degree or in-degree.

        :param outDegree: whether to use the out-degree for the computation of the degree matrix
        :type outDegree: :class:`bool`

        :param sparse: whether to return a sparse matrix or numpy array
        :type sparse: :class:`bool`

        :returns:  A laplacian adjacency matrix.
        """
        A = self.nativeAdjacencyMatrix()
        L = self.weightMatrixType()((self.getNumVertices(), self.getNumVertices()))

        if outDegree:
            L.setdiag(self.outDegreeSequence())
        else:
            L.setdiag(self.inDegreeSequence())

        L = L - A
        
        if sparse == True:
            return L
        else:
            return L.toarray()

    #Class data 
    W = None
    vList = None
    undirected = None
    