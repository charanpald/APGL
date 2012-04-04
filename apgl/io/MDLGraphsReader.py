"""
A class to read a set of graphs in MDL format, the vertex is labelled according
to the atom type.
"""

#import io
import numpy 
from apgl.graph.VertexList import VertexList
from apgl.graph.SparseGraph import SparseGraph

class MDLGraphsReader():
    def __init__(self):
        self.atomDict = {}
        self.atomDict["C"] = 0
        self.atomDict["H"] = 1
        self.atomDict["N"] = 2
        self.atomDict["O"] = 3

    def readFromFile(self, fileName):
        inFile = open(fileName,"r")
        numFeatures = 1

        graphList = []
        line = inFile.readline()

        while line != "":
            #First 3 lines are useless
            
            inFile.readline()
            inFile.readline()

            #4th line has edge information
            line = inFile.readline()
            valueList = line.split(None)
            numVertices = int(valueList[0])
            #Not strictly the number of edges, as molecules can have multiple edges
            #between a pair of atoms 
            numEdges = int(valueList[1])

            vList = VertexList(numVertices, numFeatures)

            for i in range(numVertices):
                line = inFile.readline()
                valueList = line.split(None)
                vList.setVertex(i, numpy.array([self.atomDict[valueList[3]]]))

            graph = SparseGraph(vList)

            for i in range(numEdges):
                line = inFile.readline()
                valueList = line.split(None)
                graph.addEdge(int(valueList[0])-1, int(valueList[1])-1)
        
            graphList.append(graph)

            #Ignore next two lines
            inFile.readline()
            inFile.readline()
            line = inFile.readline()

        return graphList 