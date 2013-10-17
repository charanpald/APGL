
import numpy
import heapq
import scipy.sparse 
from apgl.graph.AbstractSingleGraph import AbstractSingleGraph


class DictGraph(AbstractSingleGraph):
    """
    A graph with nodes stored in a dictionary. In particular the graph data structure is a
    dict of dicts. Edges and vertices can be labeled with anything.
    """
    def __init__(self, undirected=True):
        """
        Create a new DictGraph. 

        :param undirected: Specify whether the graph has directed or undirected edges 
        :type undirected: `bool`        
        
        """
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
        
        :param edgeValues: A corresponding list of vertex values. 
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

    def __touchVertex(self, vertexId):
        """
        If the vertex exists, do nothing. Otherwise add it to vertices and
        adjacencies. 
        """
        if vertexId not in self.vertices:
            self.vertices[vertexId] = None
        if vertexId not in self.adjacencies:
            self.adjacencies[vertexId] = {}

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
        Returns the value of the edge between two vertices. If there is no edge 
        between the vertices None is returned. 

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

    def neighbours(self, vertexId):
        """
        Find a list of neighbours of the current vertex. In a directed graph, it
        is the list of all vertices with an edge from the current vertex. 

        :param vertexId: The id of the vertex.
        """
        if vertexId not in self.adjacencies:
            raise ValueError("Vertex is not present in graph: " + str(vertexId))

        return list(self.adjacencies[vertexId].keys())

    def getVertex(self, vertexId):
        """
        Returns the label of the given vertex, or None if no label.

        :param vertex: The name of the first vertex.
        """
        if vertexId not in self.vertices:
            raise ValueError("Vertex is not present in graph: " + str(vertexId))
        return self.vertices[vertexId]

    def setVertex(self, vertexId, vertex):
        """
        Sets the vertexId with the value. Overwrites value if already present. 
        """
        self.__touchVertex(vertexId)
        self.vertices[vertexId] = vertex

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


    def getSparseWeightMatrix(self, format="lil"):
        """
        Returns a weight matrix representation of the graph as a scipy sparse 
        lil_matrix by default. The indices in the matrix correspond to the keys 
        returned by getAllVertexIds. Available formats are: lil for scipy.sparse.lil_matrix, 
        csr for scipy.sparse.csr_matrix, csc for scipy.sparse.csc_matrix, and 
        pysparse for pysparse's ll_mat. 
        
        :param format: The format of the sparse matrix. 
        """
        if format=="lil": 
            W = scipy.sparse.lil_matrix((self.size, self.size))
            W = self.__populateWeightMatrix(W) 
        elif format=="csr": 
            W = scipy.sparse.lil_matrix((self.size, self.size))
            W = self.__populateWeightMatrix(W)
            W = W.tocsr()
        elif format=="csc":
            W = scipy.sparse.lil_matrix((self.size, self.size))
            W = self.__populateWeightMatrix(W)
            W = W.tocsc()
        elif format=="pysparse": 
            from pysparse import spmatrix
            W = spmatrix.ll_mat(self.size, self.size)
            W = self.__populateWeightMatrix(W)
        else:
            raise ValueError("Invalid format: " + format)
            
        return W 

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

            subgraph.vertices[vertexId] = self.vertices[vertexId]

        deleteIdList = []

        #Now remove the elements in the adjacencies:
        for vertex1 in subgraph.adjacencies.keys():
            for vertex2 in subgraph.adjacencies[vertex1].keys():
                if vertex2 not in vertexIds:
                    deleteIdList.append((vertex1, vertex2))

        for vertex1, vertex2 in deleteIdList:         
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
        degSeq = numpy.zeros(self.getNumVertices(), numpy.int)

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
        degSeq = numpy.zeros(self.getNumVertices(), numpy.int)

        vertexIds = self.getAllVertexIds()
        for i in range(len(vertexIds)):
            vertex = vertexIds[i]
            for vertex2 in self.adjacencies[vertex].items():
                degSeq[vertexIds.index(vertex2[0])] += 1
                
            vertexList.append(vertex)

        return degSeq, vertexList

    def vertexExists(self, vertexId):
        """
        Returns true if the vertex with the given name exists, otherwise false. 
        """
        return vertexId in self.vertices
        
    def edgeExists(self, vertexId1, vertexId2): 
        """
        Return true if the edge exists between two vertices
        """
        if not self.vertexExists(vertexId1): 
            return False 
        
        return vertexId2 in self.adjacencies[vertexId1]

    def removeVertex(self, vertexId): 
        """
        Remove a vertex and all its edges. 
        
        :param vertexId: The id of the vertex to remove. 
        """
        neighbours = self.neighbours(vertexId)
        del self.adjacencies[vertexId]
        del self.vertices[vertexId]

        if self.undirected:
            for vertexId2 in neighbours: 
                del self.adjacencies[vertexId2][vertexId]
        else: 
            for vertexId2 in self.getAllVertexIds(): 
                if vertexId in self.adjacencies[vertexId2]: 
                    del self.adjacencies[vertexId2][vertexId]

    def toSparseGraph(self): 
        """
        Convert the current graph to a SparseGraph. Currently, vertex labels 
        are not converted. 
        """
        from apgl.graph import SparseGraph        
        
        W = self.getSparseWeightMatrix(format="csr")
        graph = SparseGraph(W.shape[0], W=W, undirected=self.undirected)
        
        return graph 

    def toPySparseGraph(self): 
        """
        Convert the current graph to a PySparseGraph. Currently, vertex labels 
        are not converted. 
        """
        from apgl.graph import PySparseGraph        
        
        W = self.getSparseWeightMatrix(format="pysparse")
        graph = PySparseGraph(W.shape[0], W=W, undirected=self.undirected)
        
        return graph         

    def degreeSequence(self):
        """
        :returns: a vector of the degrees (including self edges) for each vertex for an undirected graph.
        """
        if not self.isUndirected():
            raise ValueError("degreeSequence is only for undirected graphs")

        degSequence = self.outDegreeSequence()[0]

        #A very slow method of adding diagonal entries
        for j, i in enumerate(self.getAllVertexIds()):
            if self.getEdge(i, i) != None:
                degSequence[j] += 1

        return degSequence 

    def getNumDirEdges(self):
        """
        :returns: the number of edges, taking this graph as a directed graph.
        """
        i = 0 
        for v in self.adjacencies.keys(): 
            i += len(self.adjacencies[v])
        
        return i

    def dijkstrasAlgorithm(self, vertexId, neighbourLists=None):
        """
        Run Dijkstras Algorithm on the graph for a given source vertex. Returns
        an array with the distance to all vertices (including itself).

        :param vertexId: the index of the source vertex.

        :returns: An array whose ith element is the distance to vertex i. 
        """
        if neighbourLists!=None:
            neighbourIndices, neighbourWeights = neighbourLists

            if len(neighbourIndices) != self.getNumVertices() or len(neighbourWeights) != self.getNumVertices():
                raise ValueError("Adjacency lists must be of same size as graph")
        else:
            neighbourIndices, neighbourWeights = self.adjacencyList()

        
        vertexIds = self.getAllVertexIds()        
        
        previous = numpy.zeros(self.size)
        #This is an array of distances so far with first col as distance, and 2nd cols 
        distance = numpy.ones((self.size, 2))*numpy.inf
        distance[vertexIds.index(vertexId), 0] = 0
        distance[:, 1] = numpy.arange(self.size)
        distance = distance.tolist()
        heapq.heapify(distance)

        #Dictionary of the tuples indexed by the vertex index returned at the end 
        distanceDict = {}
        for i in distance:
            distanceDict[i[1]] = i
        INVALID = -1
        
        #distanceArray is distance for each vertex 
        distanceArray = numpy.ones(self.size)*numpy.inf
        notVisited = numpy.ones(self.size, numpy.bool)

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
            cols = numpy.array(neighbourIndices[minVertexIndex])
            weights = numpy.array(neighbourWeights[minVertexIndex])
            #updateDistances(cols, weights, minVertexDistance, distanceDict, previous, distanceArray)
            
            #If no neighbours then move onto next vertex 
            if cols.size == 0: 
                break 

            newDistances = weights + minVertexDistance
            isBetter = numpy.logical_and(newDistances < distanceArray[cols], notVisited[cols])

            for i in range(cols[isBetter].shape[0]):
                j = cols[isBetter][i]
                distanceDict[j][1] = INVALID
                distanceDict[j] = [newDistances[isBetter][i], j]
                heapq.heappush(distance, distanceDict[j])
                distanceArray[j] = newDistances[isBetter][i]

        return distanceArray

    def adjacencyList(self):
        """
        Returns an adjacency list representation L of the graph, in which L[i]
        is the list of all neighbour *indices* of vertex index i. Furthermore, the method 
        returns W in which W[i] which is the corresponding set of weights. In essence, 
        this method is a way to map vertex ids to indices. 

        :returns L: A list whose ith element is a list of neighbours for vertex i.
        :returns W: A list whose ith element is a list of neighbour weights for vertex i.
        """
        neighbourIndices = []
        neighbourWeights = []
        vertexIds = self.getAllVertexIds()
        
        for i in vertexIds:
            neighbours = list(map(vertexIds.index, self.adjacencies[i].keys()))
            neighbourIndices.append(neighbours)
            neighbourWeights.append(list(self.adjacencies[i].values()))

        return neighbourIndices, neighbourWeights

    def findAllDistances(self, useWeights=True):
        """
        Use the repeated calls to Dijkstra'  algorithm to find the shortest path 
        between all pairs of vertices. Note that the shortest path of a vertex 
        to itself is always zero. Returns a matrix whose ij th entry is the
        shortest path between vertices i and j.

        :returns:  A matrix of shortest paths between all vertices.
        """
        neighbourLists = self.adjacencyList()
        P = numpy.zeros((self.size, self.size))

        for i, vertexId in enumerate(self.getAllVertexIds()):
            P[i, :] = self.dijkstrasAlgorithm(vertexId, neighbourLists)

        return P 

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
        newGraph.vs["label"] = self.getVertices(self.getAllVertexIds())

        vertices = self.getAllVertexIds()
        allEdges = self.getAllEdges()
        

        for i in range(len(allEdges)):
            vertexId1 = vertices.index(allEdges[i][0])
            vertexId2 = vertices.index(allEdges[i][1])            
            
            newGraph.add_edge(vertexId1, vertexId2)
            newGraph.es[i]["value"] = self.getEdge(allEdges[i][0], allEdges[i][1])

        return newGraph

    vertices = None 
    adjacencies = None 
    undirected = None
    size = property(getNumVertices, doc="The number of vertices in the graph")