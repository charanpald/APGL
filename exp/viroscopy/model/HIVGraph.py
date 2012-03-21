import numpy
import os
import zipfile
import pickle
import logging
from apgl.graph.PySparseGraph import PySparseGraph
from exp.viroscopy.model.HIVVertices import HIVVertices
from apgl.util.Parameter import Parameter
from apgl.util.Util import Util 

"""
A HIVGraph is a Graph except that its vertices are HIVIndividuals and
there are some useful method to find the S, I and R individuals. 
"""

class HIVGraph(PySparseGraph):
    def __init__(self, numVertices, undirected=True):
        """
        Create a graph with the specified number of vertices, and choose whether
        edges are directed. 
        """
        vList = HIVVertices(numVertices)
        super(HIVGraph, self).__init__(vList, undirected, sizeHint=10000)

    def getSusceptibleSet(self):
        V = self.vList.getVertices(list(range(self.getNumVertices())))
        susceptibleSet = numpy.nonzero(V[:, HIVVertices.stateIndex] == HIVVertices.susceptible)[0]
        return set(susceptibleSet.tolist()) 

    def getInfectedSet(self):
        V = self.vList.getVertices(list(range(self.getNumVertices())))
        infectedSet = numpy.nonzero(V[:, HIVVertices.stateIndex] == HIVVertices.infected)[0]
        return set(infectedSet.tolist())

    def getRemovedSet(self):
        V = self.vList.getVertices(list(range(self.getNumVertices())))
        removedSet = numpy.nonzero(V[:, HIVVertices.stateIndex] == HIVVertices.removed)[0]
        return set(removedSet.tolist())

    def setRandomInfected(self, numInitialInfected, t=0.0):
        Parameter.checkInt(numInitialInfected, 0, self.getNumVertices())
        inds = numpy.random.permutation(numInitialInfected)

        for i in inds[0:numInitialInfected]:
            self.getVertexList().setInfected(i, t)

    def detectedNeighbours(self, vertexInd):
        """
        Return an array of the detected neighbours.
        """
        V = self.vList.getVertices(list(range(self.getNumVertices())))
        neighbours = self.neighbours(vertexInd)
        return neighbours[V[neighbours, HIVVertices.stateIndex] == HIVVertices.removed]

    @classmethod
    def load(cls, filename, vListType=None):
        """
        Load the graph object from the corresponding file. Data is loaded in a zip
        format as created using save().

        :param filename: The name of the file to load.
        :type filename: :class:`str`

        :returns: A graph corresponding to the one saved in filename.
        """
        Parameter.checkClass(filename, str)

        (path, filename) = os.path.split(filename)
        if path == "":
            path = "./"

        originalPath = os.getcwd()
        try:
            os.chdir(path)

            myzip = zipfile.ZipFile(filename + '.zip', 'r')
            myzip.extractall()
            myzip.close()

            #Deal with legacy files
            try:
                W = cls.loadMatrix(cls.wFilename)
                metaDict = Util.loadPickle(cls.metaFilename)
                if vListType == None:
                    vList = globals()[metaDict["vListType"]].load(cls.verticesFilename)
                else:
                    vList = vListType.load(cls.verticesFilename)
                undirected = metaDict["undirected"]

            except IOError:
                W = cls.loadMatrix(filename + cls.matExt)
                vList = VertexList.load(filename)
                undirected = Util.loadPickle(filename + cls.boolExt)

            graph = cls(vList.getNumVertices(), undirected)
            graph.W = W
            graph.setVertexList(vList)

            for tempFile in myzip.namelist():
                os.remove(tempFile)
        finally:
            os.chdir(originalPath)

        logging.debug("Loaded graph from file " + filename)
        return graph

    def infectedIndsAt(self, t): 
        """
        Compute the indices of this graph consisting only of all vertices 
        infected before time t
        
        :param t: The time point to consider. 
        :type t: `float`
        """
        
        vertexArray = self.getVertexList().getVertices()
        inds = numpy.arange(self.getNumVertices())[numpy.logical_and(vertexArray[:, HIVVertices.infectionTimeIndex] <= t, vertexArray[:, HIVVertices.infectionTimeIndex] >= 0)]
        
        return inds
        