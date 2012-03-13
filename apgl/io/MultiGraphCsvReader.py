
from apgl.io.CsvReader import CsvReader
from apgl.graph.VertexList import VertexList
from apgl.graph.SparseMultiGraph import SparseMultiGraph 
import numpy
import logging

"""
Note that this is not a GraphReader since it does not read from a single file
but multiple files. 
"""

class MultiGraphCsvReader(CsvReader):
    def __init__(self, idIndex, featureIndices, converters, nanProcessor=None):
        self.idIndex = idIndex
        self.featureIndices = featureIndices
        self.converters = converters
        featureIndices.insert(0, idIndex)
        self.vertexIndices = tuple(featureIndices)
        self.nanProcessor = nanProcessor 

    def readGraph(self, vertexFileName, edgeFileNames, undirected=True, delimiter=None):
        """
        Read a MultiGraph from at least 2 files: one is the information about
        vertices and the other(s) are lists of edges. For the list of vertices
        the first column must be the ID of the vertex. 
        """
        
        X = numpy.loadtxt(vertexFileName, skiprows=1, converters=self.converters, usecols=self.vertexIndices, delimiter=delimiter)

        numVertices = X.shape[0]
        numFeatures = X.shape[1]-1 

        vertexIds = X[:, 0]
        vertexIdsDict = {}

        for i in range(0, numVertices):
            vertexIdsDict[vertexIds[i]] = i

        if self.nanProcessor != None:
            X[:, 1:numFeatures+1] = self.nanProcessor(X[:, 1:numFeatures+1])

        vertexList = VertexList(numVertices, numFeatures)
        vertexList.setVertices(X[:, 1:numFeatures+1])
        
        maxEdgeTypes = len(edgeFileNames)
        sparseMultiGraph = SparseMultiGraph(vertexList, maxEdgeTypes, undirected)

        for i in range(0, maxEdgeTypes):
            self.__readEdgeFile(vertexIdsDict, edgeFileNames[i], sparseMultiGraph, i)

        logging.info("MultiGraph read with " + str(sparseMultiGraph.getNumVertices()) + " vertices and " + str(sparseMultiGraph.getNumEdges()) + " edges")

        return sparseMultiGraph

    def __readEdgeFile(self, vertexIdsDict, edgeFileName, sparseMultiGraph, edgeType):
        """
        Each edge file contains a list of edges with possible weights.
        """
        edges = numpy.loadtxt(edgeFileName)
        numEdges = edges.shape[0]

        if edges.shape[1] == 2:
            for i in range(0, numEdges):
                sparseMultiGraph.addEdge(vertexIdsDict[edges[i, 0]], vertexIdsDict[edges[i, 1]], edgeType)
        elif edges.shape[1] == 3:
            for i in range(0, numEdges):
                sparseMultiGraph.addEdge(vertexIdsDict[edges[i, 0]], vertexIdsDict[edges[i, 1]], edgeType, edges[i, 2])
        else:
            raise ValueError("Bad edge file")

    idIndex = None
    featureIndices = None
    converters = None