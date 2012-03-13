'''
Created on 24 Jul 2009

@author: charanpal
'''

from apgl.data.ExamplesList import ExamplesList
from apgl.graph.VertexList import VertexList
from apgl.graph.SparseGraph import SparseGraph
import numpy 

class EgoUtils():
    def __init__(self):
        pass
    
    @staticmethod
    def graphFromMatFile(matFileName):
        """
        Generate a sparse graph from a Matlab file of ego and alters and their transmissions. This is a mostly 
        disconnected graph made up of pairs of connected vertices, i.e each vertex has degree 1.  
        """
        examplesList = ExamplesList.readFromMatFile(matFileName)
        numExamples = examplesList.getNumExamples()
        numFeatures = examplesList.getDataFieldSize("X", 1)
        numVertexFeatures = numFeatures/2+1
        vList = VertexList(numExamples*2, int(numVertexFeatures))
        sGraph = SparseGraph(vList)
        
        for i in range(0, examplesList.getNumExamples()): 
            v1Index = i*2 
            v2Index = i*2+1
            example = examplesList.getSubDataField("X", numpy.array([i])).ravel()
            vertex1 = numpy.r_[example[0:numFeatures/2], numpy.array([1])]
            vertex2 = numpy.r_[example[numFeatures/2:numFeatures], numpy.array([0])]
            
            sGraph.setVertex(v1Index, vertex1)
            sGraph.setVertex(v2Index, vertex2)
            sGraph.addEdge(v1Index, v2Index)
        
        return sGraph 
    
    @staticmethod
    def getTotalInformation(graph):
        totalInfo = 0 
        numVertices = graph.getNumVertices()
        numFeatures = graph.getVertexList().getNumFeatures()
        infoIndex = numFeatures-1

        for i in range(numVertices): 
            vertex = graph.getVertex(i)
            if vertex[infoIndex] == 1:
                totalInfo = totalInfo + 1 
        
        return totalInfo

    @staticmethod 
    def averageHopDistance(transmissions):
        """
        Take a list of numpy arrays which has rows as an information transmission.
        Outputs the total number of hops of information divided by the total
        number of original senders (those that did not receive from another person).
        A measure of the average spread of information from each source. 
        """
        numIterations = len(transmissions)
        originalInfoSenders = numpy.array([])
        infoReceivers = numpy.array([])
        totalHops = 0

        #Assume transmissions are unique 
        for i in range(0, numIterations):
            currentAlters = transmissions[i][:, 1]
            infoReceivers = numpy.union1d(infoReceivers, currentAlters)
            totalHops += transmissions[i].shape[0]

            currentEgos = transmissions[i][:, 0]
            originalInfoSenders = numpy.union1d(originalInfoSenders, currentEgos)
            originalInfoSenders = numpy.setdiff1d(originalInfoSenders, infoReceivers)

        #Number of path ends is infoReceivers.shape[0]
        if originalInfoSenders.shape[0] != 0:
            return float(totalHops)/originalInfoSenders.shape[0]
        else:
            return 0 

    @staticmethod
    def receiversPerSender(transmissions):
        """
        Take a list of numpy arrays which has rows as an information transmission.
        Outputs the total number of receivers divided by the total number of senders.
        A measure of the number of people each sender transmits the information to. 
        """
        numIterations = len(transmissions)
        senders = numpy.array([])
        receivers = numpy.array([])

        #Assume transmissions are unique
        for i in range(0, numIterations):
            currentAlters = transmissions[i][:, 1]
            receivers = numpy.union1d(receivers, currentAlters)

            currentEgos = transmissions[i][:, 0]
            senders = numpy.union1d(senders, currentEgos)

        if senders.shape[0] == 0:
            return 0
        else:
            return float(receivers.shape[0])/senders.shape[0]
    


        
                    

        
