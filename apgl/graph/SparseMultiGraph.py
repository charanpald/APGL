'''
Represents a sparse multigraph, which can be directed or undirected, and has weights
on the edges.  The list of vertices is immutable (see VertexList), however edges
can be added or removed. The graph is efficient in memory usage if there are a
sparse set of edges. Note that you cannot add edges with zero weights.
Created on 3 Feb 2010 

@author: charanpal
'''
from apgl.graph.AbstractMultiGraph import AbstractMultiGraph
from apgl.graph.SparseGraph import SparseGraph
from apgl.util.Parameter import Parameter
import numpy

class SparseMultiGraph(AbstractMultiGraph):
    def __init__(self, vList, maxEdgeTypes, undirected=True):
        Parameter.checkInt(maxEdgeTypes, 1, float('inf'))
        self.vList = vList
        self.undirected = undirected
        self.maxEdgeTypes = maxEdgeTypes

        self.sparseGraphs = []

        for i in range(maxEdgeTypes):
            self.sparseGraphs.append(SparseGraph(vList, undirected))

    def addEdge(self, vertexIndex1, vertexIndex2, edgeTypeIndex, edge=1):
        """ Add an edge to the graph between two vertices.

        @param vertexIndex1: The index of the first vertex.
        @param vertexIndex1: The index of the second vertex.
        @param edge: The value to assign to the edge.
        """
        Parameter.checkIndex(edgeTypeIndex, 0, self.maxEdgeTypes)
        self.sparseGraphs[edgeTypeIndex].addEdge(vertexIndex1, vertexIndex2, edge)

    def removeEdge(self, vertexIndex1, vertexIndex2, edgeTypeIndex):
        """ Remove an edge between two vertices.

        @param vertexIndex1: The index of the first vertex.
        @param vertexIndex1: The index of the second vertex.
        """
        Parameter.checkIndex(edgeTypeIndex, 0, self.maxEdgeTypes)
        self.sparseGraphs[edgeTypeIndex].removeEdge(vertexIndex1, vertexIndex2)

    def getNumEdges(self, edgeTypeIndex=-1):
        """
        Returns the total number of edges in the graph of a given type. If the
        edgeType is -1 then returns the total number of indices of all types. 
        """
        Parameter.checkIndex(edgeTypeIndex, -1, self.maxEdgeTypes)

        if edgeTypeIndex == -1:
            numEdges = 0
            for i in range(0, self.maxEdgeTypes):
                numEdges = numEdges + self.sparseGraphs[i].getNumEdges()
        else:
            numEdges = self.sparseGraphs[edgeTypeIndex].getNumEdges()

        return numEdges 

    def getNumVertices(self):
        """ Returns the total number of vertices in the graph. """
        return self.vList.getNumVertices()

    def isUndirected(self):
        """ Returns true if the current graph is undirected, otherwise false. """
        return self.undirected

    def getNeighboursByEdgeType(self, vertexIndex1, edgeTypeIndex):
        """ Return a iterable item of neighbours (indices) """
        return list(self.sparseGraphs[edgeTypeIndex].neighbours(vertexIndex1))

    def neighbours(self, vertexIndex1):
        """ Return a list of all neighbours """ 
        neighbours = []

        for i in range(0, self.maxEdgeTypes):
            for v in self.sparseGraphs[i].neighbours(vertexIndex1):
                neighbours.append(v)

        return list(set(neighbours))

    def getEdge(self, vertexIndex1, vertexIndex2, edgeTypeIndex):
        """ Return an edge between two vertices """
        return self.sparseGraphs[edgeTypeIndex].getEdge(vertexIndex1, vertexIndex2)

    def getVertex(self, vertexIndex):
        """ Return a vertex of given index """
        return self.vList.getVertex(vertexIndex)

    def getAllVertexIds(self):
        """ Return all indices of the vertices """
        return list(range(0, self.vList.getNumVertices()))

    def setVertex(self, vertexIndex, vertex):
        """ Assign a value to a vertex """
        self.vList.setVertex(vertexIndex, vertex)

    def getAllEdges(self):
        """
        Return an array of edges with each row representing an edge and its type index.
        """
        allEdges = numpy.zeros((0, 3))

        for i in range(0, self.maxEdgeTypes):
            edges = self.sparseGraphs[i].getAllEdges()
            edges = numpy.c_[edges, numpy.ones(edges.shape[0])*i]
            allEdges = numpy.r_[allEdges, edges]

        return allEdges

    def getSparseGraph(self, edgeTypeIndex):
        return self.sparseGraphs[edgeTypeIndex]


    def getVertexList(self):
        return self.vList