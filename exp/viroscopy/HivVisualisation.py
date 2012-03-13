
from apgl.viroscopy.HIVGraphReader import HIVGraphReader
from apgl.io.PajekWriter import PajekWriter
from apgl.util import * 
import logging
import sys
import math 


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

"""
A script to load up the HIV data in its entirity.
"""
hivReader = HIVGraphReader()
graph = hivReader.readHIVGraph()

def getVertexSize(vertexIndex, graph):
    return math.sqrt(len(graph.neighbours(vertexIndex)))

def getEdgeWeight(vertexIndex1, vertexIndex2, graph):
    return graph.getEdge(vertexIndex1, vertexIndex2, 0)

outputDirectory = PathDefaults.getOutputDir()
fileName = outputDirectory + "hivGraph"

pajekWriter = PajekWriter()
pajekWriter.setVertexSizeFunction(getVertexSize)
pajekWriter.setEdgeWeightFunction(getEdgeWeight)
pajekWriter.writeToFile(fileName, graph)