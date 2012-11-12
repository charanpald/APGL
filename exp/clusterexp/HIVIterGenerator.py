
import logging
import numpy
from apgl.graph import *
from exp.viroscopy.HIVGraphReader import HIVGraphReader
from apgl.util.DateUtils import DateUtils
from exp.sandbox.GraphIterators import IncreasingSubgraphListIterator


class HIVIterGenerator(object): 
    def __init__(self, minGraphSize=500, monthStep=1): 

        startYear = 1900
        daysInMonth = 30      
        
        #Start off with the HIV data 
        hivReader = HIVGraphReader()
        graph = hivReader.readHIVGraph()
        fInds = hivReader.getIndicatorFeatureIndices()
        
        #The set of edges indexed by zeros is the contact graph
        #The ones indexed by 1 is the infection graph
        edgeTypeIndex1 = 0
        edgeTypeIndex2 = 1
        sGraphContact = graph.getSparseGraph(edgeTypeIndex1)
        sGraphInfect = graph.getSparseGraph(edgeTypeIndex2)
        sGraphContact = sGraphContact.union(sGraphInfect)
        graph = sGraphContact
        
        #Find max component
        #Create a graph starting from the oldest point in the largest component 
        components = graph.findConnectedComponents()
        graph = graph.subgraph(list(components[0]))
        logging.debug(graph)
        
        detectionIndex = fInds["detectDate"]
        vertexArray = graph.getVertexList().getVertices()
        detections = vertexArray[:, detectionIndex]
        
        firstVertex = numpy.argmin(detections)
        
        dayList = list(range(int(numpy.min(detections)), int(numpy.max(detections)), daysInMonth*monthStep))
        dayList.append(numpy.max(detections))
        
        subgraphIndicesList = []
        
        #Generate subgraph indices list 
        for i in dayList:
            logging.info("Date: " + str(DateUtils.getDateStrFromDay(i, startYear)))
            subgraphIndices = numpy.nonzero(detections <= i)[0]
            
            #Check subgraphIndices are sorted 
            subgraphIndices = numpy.sort(subgraphIndices)
            currentSubgraph = graph.subgraph(subgraphIndices)
            compIndices = currentSubgraph.depthFirstSearch(list(subgraphIndices).index(firstVertex))
            subgraphIndices =  subgraphIndices[compIndices]
            
            if subgraphIndices.shape[0] >= minGraphSize: 
                subgraphIndicesList.append(subgraphIndices)
        
        self.graph = graph
        self.subgraphIndicesList = subgraphIndicesList
        
        self.numGraphs = len(subgraphIndicesList)
        
    def getNumGraphs(self): 
        return self.numGraphs 
    
    def getIterator(self):
        return IncreasingSubgraphListIterator(self.graph, self.subgraphIndicesList)