'''
Created on 23 Jul 2009

@author: charanpal
'''
from apgl.data.ExamplesList import ExamplesList
from apgl.graph.DictGraph import DictGraph
from apgl.graph.SparseGraph import SparseGraph
from apgl.util.Parameter import Parameter
import numpy

class EgoSimulator:
    """
    A class which simulates information diffusion processes within an Ego Network. In
    this case one models information by taking as input a graph and then iterating using
    a classifier for all pairs of vertices. The presence of information is stored
    in the last index of the VertexList as a (0, 1) value. 
    """
    def __init__(self, graph, egoPairClassifier, preprocessor=None):
        self.graph = graph 
        self.egoPairClassifier = egoPairClassifier
        self.preprocessor = preprocessor 
        self.iteration = 0 
        self.allTransmissionEdges = []
        self.transmissionGraph = DictGraph(False)

        self.numVertexFeatures = self.graph.getVertexList().getNumFeatures()
        self.numPersonFeatures = self.numVertexFeatures-1
        self.infoIndex = self.numVertexFeatures-1
        self.edges = self.graph.getAllEdges()
        
    def advanceGraph(self):
        #First, find all the edges in the graph and create an ExampleList      
        blockSize = 5000
        
        X = numpy.zeros((blockSize, self.numPersonFeatures*2))
        possibleTransmissionEdges = []
        possibleTransmissionEdgeIndices = []
        
        for i in range(self.edges.shape[0]):
            vertex1 = self.graph.getVertex(self.edges[i,0])
            vertex2 = self.graph.getVertex(self.edges[i,1])

            if vertex1[self.infoIndex] == 1 and vertex2[self.infoIndex] == 0:
                X[len(possibleTransmissionEdges), :] = numpy.r_[vertex1[0:self.numPersonFeatures], vertex2[0:self.numPersonFeatures]]
                possibleTransmissionEdges.append((self.edges[i,0], self.edges[i,1]))
                possibleTransmissionEdgeIndices.append(i)
            if vertex2[self.infoIndex] == 1 and vertex1[self.infoIndex] == 0:
                X[len(possibleTransmissionEdges), :] = numpy.r_[vertex2[0:self.numPersonFeatures], vertex1[0:self.numPersonFeatures]]
                possibleTransmissionEdges.append((self.edges[i,1], self.edges[i,0]))
                possibleTransmissionEdgeIndices.append(i)

            #Increase X if it is small 
            if (len(possibleTransmissionEdges) == X.shape[0]):
                X = numpy.r_[X, numpy.zeros((blockSize, self.numPersonFeatures*2))]

        #Now, remove from edges the ones that can possible have a transmission
        self.edges = numpy.delete(self.edges, possibleTransmissionEdgeIndices, 0)
        X = X[0:len(possibleTransmissionEdges), :]

        name = "X"
        examplesList = ExamplesList(X.shape[0])
        examplesList.addDataField(name, X)
        examplesList.setDefaultExamplesName(name)
        
        if self.preprocessor != None: 
            X = examplesList.getDataField(examplesList.getDefaultExamplesName())
            X = self.preprocessor.standardiseArray(X) 
            examplesList.overwriteDataField(examplesList.getDefaultExamplesName(), X)
        
        y = self.egoPairClassifier.classify(X)
        
        transmissionEdges = numpy.zeros((sum(y==1), 2), numpy.int)
        j = 0 

        #Now, update the vertices to reflect transfer 
        for i in range(len(possibleTransmissionEdges)):
            if y[i] == 1:
                transmissionEdges[j, 0] = possibleTransmissionEdges[i][0]
                transmissionEdges[j, 1] = possibleTransmissionEdges[i][1]

                self.transmissionGraph.setVertex(int(transmissionEdges[j, 0]), self.graph.getVertex(transmissionEdges[j, 0]))
                self.transmissionGraph.setVertex(int(transmissionEdges[j, 1]), self.graph.getVertex(transmissionEdges[j, 1]))
                self.transmissionGraph.addEdge(int(transmissionEdges[j, 0]), int(transmissionEdges[j, 1]), 1)

                j += 1

                vertex = self.graph.getVertex(possibleTransmissionEdges[i][1])
                vertex[self.infoIndex] = 1
                self.graph.setVertex(possibleTransmissionEdges[i][1], vertex)
                    
        self.allTransmissionEdges.append(transmissionEdges)
        self.iteration = self.iteration + 1
        
        return self.graph


    def fullTransGraph(self):
        """
        This function will return a new graph which contains a directed edge if
        a transmission will occur between two vertices. 
        """
        if self.iteration != 0:
            raise ValueError("Must run fullTransGraph before advanceGraph")

        #First, find all the edges in the graph and create an ExampleList
        numEdges = self.edges.shape[0]
        X = numpy.zeros((numEdges*2, self.numPersonFeatures*2))
        ind = 0 

        for i in range(numEdges):
            vertex1 = self.graph.getVertex(self.edges[i,0])
            vertex2 = self.graph.getVertex(self.edges[i,1])

            X[ind, :] = numpy.r_[vertex1[0:self.numPersonFeatures], vertex2[0:self.numPersonFeatures]]
            X[ind+numEdges, :] = numpy.r_[vertex2[0:self.numPersonFeatures], vertex1[0:self.numPersonFeatures]]
            ind = ind + 1

        name = "X"
        examplesList = ExamplesList(X.shape[0])
        examplesList.addDataField(name, X)
        examplesList.setDefaultExamplesName(name)

        if self.preprocessor != None:
            X = self.preprocessor.process(examplesList.getDataField(examplesList.getDefaultExamplesName()))
            examplesList.overwriteDataField(examplesList.getDefaultExamplesName(), X)

        y = self.egoPairClassifier.classify(examplesList.getSampledDataField(name))
        fullTransmissionGraph = SparseGraph(self.graph.getVertexList(), False)

        transIndices = numpy.nonzero(y==1)[0]

        #Now, write out the transmission graph 
        for i in range(len(transIndices)):
            if transIndices[i] < numEdges:
                fullTransmissionGraph.addEdge(self.edges[transIndices[i],0], self.edges[transIndices[i],1])
            else:
                fullTransmissionGraph.addEdge(self.edges[transIndices[i]-numEdges,1], self.edges[transIndices[i]-numEdges,0])

        return fullTransmissionGraph

    def getGraph(self):
        return self.graph
    
    def getTransmissions(self, i=None):
        if i != None:
            Parameter.checkIndex(i, 0, self.iteration)
            return self.allTransmissionEdges[i]
        else:
            return self.allTransmissionEdges

    def getAlters(self, i):
        Parameter.checkIndex(i, 0, self.iteration)

        alters = []
        for j in range(0, len(self.allTransmissionEdges[i])):
            alters.append(self.allTransmissionEdges[i][j, 1])

        return numpy.unique(numpy.array(alters))
    
    def getNumIterations(self):
        return self.iteration

    def getTransmissionGraph(self):
        return self.transmissionGraph
    
    graph = None
    egoPairClassifier = None
    preprocessor = None
    iteration = None
    allTransmissionEdges = []
    transmissionGraph = None 