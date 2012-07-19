'''
Created on 30 Jun 2009

@author: charanpal
'''

import numpy 
from apgl.util.Util import Util
from apgl.graph.AbstractGraph import AbstractGraph

class AbstractSingleGraph(AbstractGraph):
    """
    A basic abstract base class for graphs. An AbstractSingleGraph can only have
    a single edge at most between any two vertices, but edges and vertices can be
    labelled with any object. Verticies are indexed with integer values.
    """
    def __init__(self):
        pass

    def addEdge(self, vertexIndex1, vertexIndex2, edgeValue):
        """
        Add a non-zero edge between two vertices.
        
        :param vertexIndex1: The index of the first vertex.
        :type vertexIndex1: :class:`int`

        :param vertexIndex2: The index of the second vertex.
        :type vertexIndex2: :class:`int`

        :param edgeValue: The value to assign to the edge.
        """

        Util.abstract()

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

        if edgeValues != []: 
            for i in range(edgeIndexArray.shape[0]):
                self.addEdge(edgeIndexArray[i, 0], edgeIndexArray[i, 1], edgeValues[i])
        else:
            for i in range(edgeIndexArray.shape[0]):
                self.addEdge(edgeIndexArray[i, 0], edgeIndexArray[i, 1], 1)
    
    def removeEdge(self, vertexIndex1, vertexIndex2):
        """
        Remove an edge between two vertices.

        :param vertexIndex1: The index of the first vertex.
        :type vertexIndex1: :class:`int`

        :param vertexIndex2: The index of the second vertex.
        :type vertexIndex2: :class:`int`
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

    def getEdge(self, vertexIndex1, vertexIndex2):
        """ Return an edge between two vertices """
        Util.abstract()

    def getVertex(self, vertexIndex):
        """ Return a vertex of given index """
        Util.abstract()

    def getVertices(self, vertexIndices):
        """
        Takes a list of vertex indices and returns the corresponding vertex values.

        :param vertexIndices: A list of vertex indices
        :type vertexIndices: :class:`list`

        :returns: A list of vertices corresponding to the indices 
        """
        vertices = [] 

        for vertexIndex in vertexIndices:
            vertices.append(self.getVertex(vertexIndex))

        return vertices 

    def getAllVertexIds(self):
        """
        Return a list of all indices of the vertices
        
        :returns:  A numpy array of all the vertex indices in this graph. 
        """

        Util.abstract()

    def setVertices(self, vertexIndices, vertices):
        """
        Assign new values to the vertices corresponding to vertexIndices

        :param vertexIndices: A list of vertex indices
        :type vertexIndices: :class:`list`

        :param vertices: A list of vertex values
        :type vertices: :class:`list`
        """
        if len(vertexIndices) != len(vertices):
            raise ValueError("Must be lists of same length")

        for i in range(len(vertexIndices)):
            self.setVertex(vertexIndices[i], vertices[i])

    def setVertex(self, vertexIndex, vertex):
        """ Assign a value to a vertex """
        Util.abstract()

    def getAllEdges(self):
        """
        Return an array of edges with each row representing an edge.

        :returns:  A numpy array of all edges in this graph. 
        """
        Util.abstract()

    def toNetworkXGraph(self):
        """
        Convert this graph into a networkx Graph or DiGraph object, which requires 
        networkx to be installed. Notice that the edge value must be hashable,
        which is the case for AbstractMatrixGraph subclasses. Edge values are stored 
        under the "value" index. Vertices are stored as indices with a "label"
        value being the corresponding vertex value.

        :returns:  A networkx Graph or DiGraph object.
        """
        try:
            import networkx
        except ImportError:
            raise ImportError("toNetworkXGraph() requires networkx")

        if self.isUndirected():
            networkXGraph = networkx.Graph()
        else:
            networkXGraph = networkx.DiGraph()

        for i in range(self.getNumVertices()):
            networkXGraph.add_node(i, label=self.getVertex(i))

        allEdges = self.getAllEdges()

        for i in range(allEdges.shape[0]):
            edgeVal = self.getEdge(allEdges[i, 0], allEdges[i, 1])
            networkXGraph.add_edge(allEdges[i, 0], allEdges[i, 1], value=edgeVal)

        return networkXGraph 

    def findConnectedComponents(self):
        """
        Finds a list of all connected components of the graph, in order of size
        with the smallest first.

        :returns: A list of lists of component indices. 
        """
        if not self.isUndirected():
            raise ValueError("Can only find components on undirected graphs")

        vertexIds = set(self.getAllVertexIds())

        components = []

        while len(vertexIds) != 0:
            currentVertex = vertexIds.pop()
            currentComponent = self.depthFirstSearch(currentVertex)
            components.append(currentComponent)
            vertexIds = vertexIds.difference(currentComponent)

        sortedIndices = numpy.array([len(x) for x in components]).argsort()
        sortedComponents = []

        for i in reversed(list(range(len(components)))):
            sortedComponents.append(components[sortedIndices[i]])

        return sortedComponents


    def __getitem__(self, vertexIndices):
        """
        This is called when using square bracket notation and returns the value
        of the specified edge, e.g. graph[i, j] returns the edge between i and j.

        :param vertexIndices: a tuple of vertex indices (i, j)
        :type vertexIndices: :class:`tuple`

        :returns: The value of the edge.
        """
        vertexIndex1, vertexIndex2 = vertexIndices
        return self.getEdge(vertexIndex1, vertexIndex2)

    def __setitem__(self, vertexIndices, value):
        """
        This is called when using square bracket notation and sets the value
        of the specified edge, e.g. graph[i, j] = 1.

        :param vertexIndices: a tuple of vertex indices (i, j)
        :type vertexIndices: :class:`tuple`

        :param value: the value of the edge
        """
        vertexIndex1, vertexIndex2 = vertexIndices
        self.addEdge(vertexIndex1, vertexIndex2, value)


    def toIGraph(self):
        """
        Convert this graph into a igraph Graph object, which requires igraph to be
        installed. Edge values are stored under the "value" index. Vertices
        are stored as indices with a "label" value being the corresponding vertex value.

        :returns:  An igraph Graph object.
        """
        try:
            import igraph
        except ImportError:
            raise ImportError("toIGraph() requires igraph")

        newGraph = igraph.Graph(self.getNumVertices(), directed= not self.isUndirected())

        #Add all vertices 
        newGraph.vs["label"] = self.getVertices(range(self.getNumVertices()))

        allEdges = self.getAllEdges()

        for i in range(allEdges.shape[0]):
            edgeVal = self.getEdge(allEdges[i, 0], allEdges[i, 1])
            newGraph.add_edges((int(allEdges[i, 0]), int(allEdges[i, 1])))
            newGraph.es[i]["value"] = edgeVal

        return newGraph


    def __str__(self):
        output = str(self.__class__.__name__) + ": "
        output += "vertices " + str(self.getNumVertices()) + ", edges " + str(self.getNumEdges())
        if self.undirected:
            output += ", undirected"
        else:
            output += ", directed"
        return output
        
    
    