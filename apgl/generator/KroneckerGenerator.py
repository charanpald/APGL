
import numpy
from apgl.util.Parameter import Parameter
from apgl.graph.VertexList import VertexList
from apgl.graph.SparseGraph import SparseGraph

class KroneckerGenerator(object):
    '''
    A class which generates graphs according to the Kronecker method.
    '''
    def __init__(self, initialGraph, k):
        """
        Initialise with a starting graph, and number of iterations k. Note that
        the starting graph must have self edges on every vertex. Only the
        adjacency matrix of the starting graph is used. 

        :param initialGraph: The intial graph to use.
        :type initialGraph: :class:`apgl.graph.AbstractMatrixGraph`

        :param k: The number of iterations.
        :type k: :class:`int`
        """
        Parameter.checkInt(k, 1, float('inf'))
        
        W = initialGraph.getWeightMatrix()
        if (numpy.diag(W)==numpy.zeros(W.shape[0])).any():
            raise ValueError("Initial graph must have all self-edges")

        self.initialGraph = initialGraph
        self.k = k

    def setK(self, k):
        """
        Set the number of iterations k.

        :param k: The number of iterations.
        :type k: :class:`int`
        """
        Parameter.checkInt(k, 1, float('inf'))
        self.k = k 

    def generate(self):
        """
        Generate a Kronecker graph using the adjacency matrix of the input graph.

        :returns: The generate graph as a SparseGraph object.
        """
        W = self.initialGraph.adjacencyMatrix()
        Wi = W

        for i in range(1, self.k):
            Wi = numpy.kron(Wi, W)

        vList = VertexList(Wi.shape[0], 0)
        graph = SparseGraph(vList, self.initialGraph.isUndirected())
        graph.setWeightMatrix(Wi)

        return graph