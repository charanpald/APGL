
from apgl.io.GraphReader import GraphReader
from apgl.io.CsvReader import CsvReader
from apgl.graph.VertexList import VertexList
from apgl.graph.SparseGraph import SparseGraph
from apgl.util.Parameter import Parameter 
import logging
import numpy
import copy 

"""
Reader a list of edges with 1 per line in CSV format (numerical, non-missing).
The first line of the CSV file is the titles. A single file contains the vertex
and edge information. 
"""

class CsvGraphReader(GraphReader, CsvReader):
    def __init__(self, vertex1Indices, vertex2Indices, converters, undirected=True):
        """
        vertex1Indices is a list of fields for the first vertex, with the 1st index
        being the ID. 
        """
        if len(vertex1Indices) < 1 or len(vertex1Indices) < 1:
            raise ValueError("vertexIndices must have at least 1 index")
        if len(vertex1Indices) != len(vertex2Indices):
            raise ValueError("len(vertex1Indices)=" + str(len(vertex1Indices)) + "and len(vertex2Indices)=" + len(vertex2Indices))

        Parameter.checkList(vertex1Indices, Parameter.checkInt, [0, float('inf')])
        Parameter.checkList(vertex2Indices, Parameter.checkInt, [0, float('inf')])

        self.vertex1IdIndex = vertex1Indices[0]
        self.vertex2IdIndex = vertex2Indices[0]



        self.vertex1Indices = copy.copy(vertex1Indices)
        self.vertex2Indices = copy.copy(vertex2Indices)
        self.vertex1Indices.remove(self.vertex1IdIndex)
        self.vertex2Indices.remove(self.vertex2IdIndex)
        self.converters = converters
        self.undirected = undirected
        self.edgeWeight = 1

    def readFromFile(self, fileName):
        X = numpy.loadtxt(fileName, skiprows=1, converters=self.converters)
        vertexIds = numpy.zeros(X.shape[0]*2)
        
        #First, we will map the vertex Ids to a set of numbers
        for i in range(0, X.shape[0]):
            vertexIds[2*i] = X[i, self.vertex1IdIndex]
            vertexIds[2*i+1] = X[i, self.vertex2IdIndex]

        vertexIds = numpy.unique(vertexIds)

        numVertices = vertexIds.shape[0]
        numFeatures = len(self.vertex1Indices)
        
        vList = VertexList(numVertices, numFeatures)
        sGraph = SparseGraph(vList, self.undirected)

        for i in range(0, X.shape[0]):
            vertex1Id = X[i, self.vertex1IdIndex]
            vertex2Id = X[i, self.vertex2IdIndex]
            
            vertex1 = X[i, self.vertex1Indices]
            vertex2 = X[i, self.vertex2Indices]

            vertex1VListId = numpy.nonzero(vertexIds==vertex1Id)[0]
            vertex2VListId = numpy.nonzero(vertexIds==vertex2Id)[0]

            vertex1VListId = int(vertex1VListId)
            vertex2VListId = int(vertex2VListId)
            vList.setVertex(vertex1VListId, vertex1)
            vList.setVertex(vertex2VListId, vertex2)

            sGraph.addEdge(vertex1VListId, vertex2VListId, self.edgeWeight)

        logging.info("Read " + fileName + " with " + str(sGraph.getNumVertices()) + " vertices and " + str(sGraph.getNumEdges()) + " edges")

        return sGraph 

    vertex1IdIndex = None
    vertex2IdIndex = None
    vertex1Indices = None
    vertex2Indices = None
    converters = None
    undirected = None
    edgeWeight = None

