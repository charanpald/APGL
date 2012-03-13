'''


Created on 03 Feb 2010

@author: charanpal
'''

from apgl.util.Util import Util
from apgl.graph.AbstractGraph import AbstractGraph

class AbstractMultiGraph(object):
    """
    A very basic abstract base class for multi-graphs. Each edge has an index so that it
    can be referred to. 
    """
    def addEdge(self, vertexIndex1, vertexIndex2, edgeTypeIndex, edge):
        """
        Add an edge to the graph between two vertices.

        @param vertexIndex1: The index of the first vertex.
        @param vertexIndex1: The index of the second vertex.
        @param edge: The value to assign to the edge.
        """

        Util.abstract()

    def removeEdge(self, vertexIndex1, vertexIndex2, edgeTypeIndex):
        """
        Remove an edge between two vertices.

        @param vertexIndex1: The index of the first vertex.
        @param vertexIndex1: The index of the second vertex.
        """
        Util.abstract()

    def getNumEdges(self):
        """ Returns the total number of edges in the graph. """
        Util.abstract()

    def getNumVertices(self):
        """ Returns the total number of vertices in the graph. """
        Util.abstract()

    def isUndirected(self):
        """ Returns true if the current graph is undirected, otherwise false. """
        Util.abstract()

    
    def neighbours(self, vertexIndex1):
        """ Return a iterable item of neighbours (indices) """
        Util.abstract()

    def getNeighboursByEdgeType(self, vertexIndex1, edgeTypeIndex):
        """ Return a iterable item of neighbours (indices) """
        Util.abstract()

    def getEdge(self, vertexIndex1, vertexIndex2, edgeTypeIndex):
        """ Return an edge between two vertices """
        Util.abstract()

    def getVertex(self, vertexIndex):
        """ Return a vertex of given index """ 
        Parameter.checkIndex(vertexIndex, 0, self.vList.getNumVertices())
        return self.vList.getVertex(vertexIndex)

    def getAllVertexIds(self):
        """ Return all indices of the vertices """ 
        Util.abstract()

    def setVertex(self, vertexIndex, vertex):
        """ Assign a value to a vertex """ 
        Util.abstract()

    def getAllEdges(self):
        """
        Return an array of edges with each row representing an edge and its type index.
        """
        Util.abstract()
