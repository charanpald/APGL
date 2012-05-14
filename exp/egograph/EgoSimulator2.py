
from exp.sandbox.data.ExamplesList import ExamplesList
from apgl.graph import * 
from apgl.util.Parameter import Parameter
import numpy

class EgoSimulator2:
    """
    A class which simulates information diffusion processes within an Ego Network. In
    this case one models information by taking as input a graph and then iterating using
    a classifier which makes predictions in the graph (as opposed to in a pairwise
    way). The presence of information is stored in the last index of the VertexList
    as a (0, 1) value.

    Note: this code is terrible, but no time to make it good. 
    """
    def __init__(self, graph, edgeLabelClassifier, preprocessor=None):
        self.graph = graph
        self.edgeLabelClassifier = edgeLabelClassifier
        self.preprocessor = preprocessor
        self.iteration = 0
        self.allTransmissionEdges = []
        self.transmissionGraph = DictGraph(False)

        self.numVertexFeatures = self.graph.getVertexList().getNumFeatures()
        self.numPersonFeatures = self.numVertexFeatures-1
        self.infoIndex = self.numVertexFeatures-1


    def advanceGraph(self):
        #First, find all the edges in the graph in which percolations can occur 
        blockSize = 5000
        edges = self.graph.getAllEdges()
        possibleTransmissionEdges = numpy.zeros((blockSize, 2), numpy.int)
        j = 0 

        for i in range(edges.shape[0]):
            vertex1 = self.graph.getVertex(edges[i,0])
            vertex2 = self.graph.getVertex(edges[i,1])

            if vertex1[self.infoIndex] == 1 and vertex2[self.infoIndex] == 0:
                possibleTransmissionEdges[j, :] = edges[i, :]
                j = j+1 
            if vertex2[self.infoIndex] == 1 and vertex1[self.infoIndex] == 0:
                inverseEdge = numpy.array([edges[i, 1], edges[i, 0]])
                possibleTransmissionEdges[j, :] = inverseEdge
                j = j+1 

            #Increase possibleTransmissionEdges if it is small
            if (j >= possibleTransmissionEdges.shape[0]-1):
                possibleTransmissionEdges = numpy.r_[possibleTransmissionEdges, numpy.zeros((blockSize, 2))]

        #Now, remove from edges the ones that can possible have a transmission
        possibleTransmissionEdges = possibleTransmissionEdges[0:j, :]

        testGraph = SparseGraph(self.graph.getVertexList())
        testGraph.addEdges(possibleTransmissionEdges, self.graph.getEdgeValues(possibleTransmissionEdges))

        y = self.edgeLabelClassifier.predictEdges(testGraph, possibleTransmissionEdges)

        #This is the list of edges where transmission did occur 
        transmissionEdges = possibleTransmissionEdges[y==1, :]

        #Now, update the vertices to reflect transfer and also record the transmission
        #graph 
        for i in range(transmissionEdges.shape[0]):
            self.transmissionGraph.setVertex(int(transmissionEdges[j, 0]), self.graph.getVertex(transmissionEdges[j, 0]))
            self.transmissionGraph.setVertex(int(transmissionEdges[j, 1]), self.graph.getVertex(transmissionEdges[j, 1]))
            self.transmissionGraph.addEdge(int(transmissionEdges[j, 0]), int(transmissionEdges[j, 1]), 1)

            vertex = self.graph.getVertex(possibleTransmissionEdges[i][1])
            vertex[self.infoIndex] = 1
            self.graph.setVertex(possibleTransmissionEdges[i][1], vertex)

        self.allTransmissionEdges.append(transmissionEdges)
        self.iteration = self.iteration + 1

        return self.graph


    def fullTransGraph(self, binaryLabels=True):
        """
        This function will return a new graph which contains a directed edge if
        a transmission will occur between two vertices.

        If binaryLabels is True then edges in the transmission graph exist only
        if the prediction of the edge label is 1. Otherwise, the transmission
        graph has identical edges to the graph given at the constructure but predicted
        labels are assigned as the values of the edges. 
        """
        if self.iteration != 0:
            raise ValueError("Must run fullTransGraph before advanceGraph")

        if self.graph.isUndirected() == False:
            raise ValueError("Expecting an undirected graph.")

        #We have to cull the final column of the vertexList which indicates if
        #information is present
        V = self.graph.getVertexList().getVertices(list(range(self.graph.getNumVertices())))
        V = V[:, 0:V.shape[1]-1]

        if self.preprocessor != None:
            V = self.preprocessor.normaliseArray(V)

        vList = VertexList(V.shape[0], V.shape[1])
        vList.setVertices(V)
        self.graph.setVertexList(vList)

        #Need to preprocess 

        #Basically, take the standard graph and classify over all edges
        #Do we want to use the directed version of the graph? Yes. 
        edges = self.graph.getAllEdges()
        inverseEdges = numpy.c_[edges[:, 1], edges[:, 0]]
        allEdges = numpy.r_[edges, inverseEdges] 

        y = self.edgeLabelClassifier.predictEdges(self.graph, allEdges)

        #y is all zeros 
        fullTransmissionGraph = SparseGraph(self.graph.getVertexList(), False)

        if binaryLabels:
            fullTransmissionGraph.addEdges(allEdges[y==1, :])
        else:
            fullTransmissionGraph.addEdges(allEdges, y)
        
        return fullTransmissionGraph