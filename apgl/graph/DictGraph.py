
import numpy
import scipy.sparse 
from apgl.graph.AbstractSingleGraph import AbstractSingleGraph

class DictGraph(AbstractSingleGraph):
    """
    A graph with nodes stored in a dictionary. In particular the graph data structure is a
    dict of dicts. Edges and vertices can be labeled with anything.
    """
    def __init__(self, undirected=True):
        self.undirected = undirected
        self.adjacencies = {}
        self.vertices = {}

    def addEdge(self, vertex1, vertex2, value=1.0):
        """
        Add an edge between two vertices.

        :param vertex1: The name of the first vertex.

        :param vertex2: The name of the second vertex.

        :param value: The value of the edge.
        """
        if value == None:
            raise ValueError("Cannot have None as edge value")

        self.__touchVertex(vertex1)
        self.__touchVertex(vertex2)
        self.adjacencies[vertex1][vertex2] = value

        if self.undirected:
           self.adjacencies[vertex2][vertex1] = value

    def addEdges(self, edgeList, edgeValues=None):
        """
        Add a set of edges to this graph given by a list of tuples of vertex ids.
        The value of each edge is simply set to 1 if edgeValues=None otherwise
        it is set to the corresponding entry of edgeValues.

        :param edgeList: A list of pairs of verted ids
        """
        i = 0
        for edge in edgeList:
            (vertex1, vertex2) = edge
            if edgeValues == None:
                value = 1
            else:
                value = edgeValues[i]

            self.__touchVertex(vertex1)
            self.__touchVertex(vertex2)
            self.adjacencies[vertex1][vertex2] = value

            if self.undirected:
                self.adjacencies[vertex2][vertex1] = value
            i += 1 

    def __touchVertex(self, vertexName):
        """
        If the vertex exists, do nothing. Otherwise add it to vertices and
        adjacencies. 
        """
        if vertexName not in self.vertices:
            self.vertices[vertexName] = None
        if vertexName not in self.adjacencies:
            self.adjacencies[vertexName] = {}

    def removeEdge(self, vertex1, vertex2):
        """
        Remove an edge. Does not remove the vertices.

        :param vertex1: The name of the first vertex.

        :param vertex2: The name of the second vertex.
        """
        self.__removeDirectedEdge(vertex1, vertex2)

        if self.undirected:
           self.__removeDirectedEdge(vertex2, vertex1)

    def __removeDirectedEdge(self, vertex1, vertex2):
        if vertex1 not in self.adjacencies:
            raise ValueError("Vertex is not present in graph: " + str(vertex1))
        if vertex2 not in self.adjacencies[vertex1]:
            raise ValueError("Vertex is not a neighbour of " + str(vertex1) + " in graph: " + str(vertex2))

        del self.adjacencies[vertex1][vertex2]

    def isUndirected(self):
        """
        Returns true if the current graph is undirected, otherwise false. 
        """
        return self.undirected

    def getNumEdges(self):
        """
        Returns the total number of edges in graph. 
        """
        numEdges = 0
        for vertex1 in self.adjacencies.keys():
            numEdges += len(self.adjacencies[vertex1])

        if not self.undirected:
            return numEdges
        else:
            #Count self edges again
            for vertex1 in self.adjacencies.keys():
                if vertex1 in self.adjacencies[vertex1]:
                    numEdges += 1
            return numEdges/2

    def getNumVertices(self):
        """
        Returns the number of vertices in the graph. 
        """
        return len(self.adjacencies)

    def getEdge(self, vertex1, vertex2):
        """
        Returns the value of the edge between two vertices.

        :param vertex1: The name of the first vertex.

        :param vertex2: The name of the second vertex.
        """
        if vertex1 not in self.adjacencies:
            raise ValueError("Vertex is not present in graph: " + str(vertex1))
        if vertex2 not in self.adjacencies:
            raise ValueError("Vertex is not present in graph: " + str(vertex2))
        
        if vertex2 not in self.adjacencies[vertex1]:
            return None 
        else:
            return self.adjacencies[vertex1][vertex2]

    def neighbours(self, vertex):
        """
        Find a list of neighbours of the current vertex. In a directed graph, it
        is the list of all vertices with an edge from the current vertex. 

        :param vertex: The name of the first vertex.
        """
        if vertex not in self.adjacencies:
            raise ValueError("Vertex is not present in graph: " + str(vertex))

        return list(self.adjacencies[vertex].keys())

    def getVertex(self, vertexName):
        """
        Returns the label of the given vertex, or None if no label.

        :param vertex: The name of the first vertex.
        """
        if vertexName not in self.vertices:
            raise ValueError("Vertex is not present in graph: " + str(vertexName))
        return self.vertices[vertexName]

    def setVertex(self, vertexName, vertex):
        """
        Sets the vertexName with the value. Overwrites value if already present. 
        """
        self.__touchVertex(vertexName)
        self.vertices[vertexName] = vertex

    def getAllVertexIds(self):
        """
        Returns a list of the vertex ids (or names) in this graph.
        """
        return list(self.vertices.keys())

    def getAllEdges(self):
        """
        Returns a list of tuples of all the edges of this graph. 
        """

        edges = []

        for vertex1 in list(self.vertices.keys()):
            for vertex2 in self.neighbours(vertex1):
                if self.undirected == True and (vertex2, vertex1) not in edges:
                    edges.append((vertex1, vertex2))
                elif self.undirected == False:
                    edges.append((vertex1, vertex2))

        return edges

    def getWeightMatrix(self):
        """
        Returns a weight matrix representation of the graph as a numpy array. The
        indices in the matrix correspond to the keys returned by getAllVertexIds.
        """
        W = numpy.zeros((self.getNumVertices(), self.getNumVertices()))
        return self.__populateWeightMatrix(W)


    def getSparseWeightMatrix(self):
        """
        Returns a weight matrix representation of the graph as a scipy sparse 
        lil_matrix. The indices in the matrix correspond to the keys returned by
        getAllVertexIds.
        """
        W = scipy.sparse.lil_matrix((self.getNumVertices(), self.getNumVertices()))
        return self.__populateWeightMatrix(W)

    def __populateWeightMatrix(self, W):
        """
        Fill the weight matrix W with edge weights according to this graph.
        """
        keys = self.vertices.keys()
        keyInds = {}

        i = 0
        for k in keys:
            keyInds[k] = i
            i += 1

        for vertex1 in keys:
            for vertex2 in self.neighbours(vertex1):
                if self.undirected == True:
                    W[keyInds[vertex1], keyInds[vertex2]] = 1
                    W[keyInds[vertex2], keyInds[vertex1]] = 1
                elif self.undirected == False:
                    W[keyInds[vertex1], keyInds[vertex2]] = 1

        return W

    def getAllEdgeIndices(self):
        """
        Returns a numpy array of size (numEdges x 2) of edge index pairs V. The ith
        row of V, V[i, :], corresponds to an edge from V[i, 0] to V[i, 1]. The corresponding
        vertex names are found using getAllVertexIds().
        """
        edges = numpy.zeros((self.getNumEdges(), 2), numpy.int)
        keyInds = {}

        i = 0
        for k in self.vertices.keys():
            keyInds[k] = i
            i += 1

        i = 0
        for vertex1 in self.vertices.keys():
            for vertex2 in self.neighbours(vertex1):
                if self.undirected and keyInds[vertex2] >= keyInds[vertex1] or not self.undirected:
                    edges[i, :] = numpy.array([keyInds[vertex1], keyInds[vertex2]])
                    i += 1 

        return edges


    def subgraph(self, vertexIds):
        """
        Compute the subgraph containing only the corresponding vertexIds and
        the edges between them. 
        """

        subgraph = DictGraph(self.undirected)

        """
        edgeList = self.getAllEdges()
        subgraphEdgeList = []

        for edge in edgeList:
            (vertex1, vertex2) = edge

            if vertex1 in vertexIds and vertex2 in vertexIds:
                subgraphEdgeList.append(edge)

        subgraph.addEdges(subgraphEdgeList)
        """
        vertexIds = set(vertexIds)

        for vertexId in vertexIds:
            if vertexId in self.adjacencies.keys():
                subgraph.__touchVertex(vertexId)
                subgraph.adjacencies[vertexId] = self.adjacencies[vertexId].copy()

        #Now remove the elements in the adjacencies:
        for vertex1 in subgraph.adjacencies.keys():
            for vertex2 in subgraph.adjacencies[vertex1].keys():
                if vertex2 not in vertexIds:
                    del subgraph.adjacencies[vertex1][vertex2]
           
        return subgraph 

    def neighbourOf(self, vertex):
        """
        Returns the list of neighbours of the current neighbour.
        """

        lst = []
        for (v1, adj) in self.adjacencies.items():
            if vertex in adj.keys():
                lst.append(v1)

        return lst

    def outDegreeSequence(self):
        """
        Find the out degree sequence. Return the sequence as a vector along with
        the corresponding vertices in a list.
        """
        vertexList = [] 
        degSeq = numpy.zeros(self.getNumVertices())

        vertexIds = self.getAllVertexIds()
        for i in range(len(vertexIds)):
            vertex = vertexIds[i]
            degSeq[i] = len(self.adjacencies[vertex])
            vertexList.append(vertex)

        return degSeq, vertexList
            
    def inDegreeSequence(self):
        """
        Find the in degree sequence. Return the sequence as a vector along with
        the corresponding vertices in a list.
        """
        vertexList = []
        degSeq = numpy.zeros(self.getNumVertices())

        vertexIds = self.getAllVertexIds()
        for i in range(len(vertexIds)):
            vertex = vertexIds[i]
            for vertex2 in self.adjacencies[vertex].items():
                degSeq[vertexIds.index(vertex2[0])] += 1
                
            vertexList.append(vertex)

        return degSeq, vertexList

    def vertexExists(self, vertex):
        """
        Returns true if the vertex with the given name exists, otherwise false. 
        """
        return vertex in self.vertices


    vertices = None 
    adjacencies = None 
    undirected = None
