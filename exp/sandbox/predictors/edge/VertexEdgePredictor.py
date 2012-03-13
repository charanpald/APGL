from apgl.graph import *
from apgl.util import *
from apgl.data.Standardiser import Standardiser
from apgl.predictors.edge.AbstractEdgePredictor import AbstractEdgePredictor
from apgl.graph.GraphUtils import GraphUtils
import numpy
import logging


class VertexEdgePredictor(AbstractEdgePredictor):
    """
    Make predictions for a new edge using the vertex labels and a learning method.
    """
    def __init__(self, learningAlg, windowSize, preprocessor=Standardiser()):
        
        self.windowSize = windowSize
        self.learningAlg = learningAlg
        self.preprocessor = preprocessor 
        self.printStep = 50 

    def learnModel(self, graph):
        """
        Take the set of pairs of edges and also non-edges and learn when an edge
        occurs. 
        """
        Parameter.checkInt(self.windowSize, 1, graph.getNumVertices())
        self.graph = graph
        X, y = GraphUtils.vertexLabelExamples(graph)

        X = self.preprocessor.learn(X)
        self.learningAlg.learnModel(X, y)


    def predictEdges(self, vertexIndices):
        """
        This makes a prediction for a series of edges using a learning algorithm
        over the vertex labels. 
        Returns a matrix with rows are a ranked list of verticies of length windowSize.
        """

        logging.info("Running predictEdges in " + str(self.__class__.__name__))

        numFeatures = self.graph.getVertexList().getNumFeatures()
        numVertices = self.graph.getNumVertices()
        

        P = numpy.zeros((vertexIndices.shape[0], self.windowSize))
        S = numpy.zeros((vertexIndices.shape[0], self.windowSize))

        #We make predictions for all edges including the self edge
        testX = numpy.zeros((numVertices, numFeatures*2))
        allVertexIndices = numpy.arange(0, numVertices)

        for i in range(vertexIndices.shape[0]):
            Util.printIteration(i, self.printStep, vertexIndices.shape[0])
            testX[:, 0:numFeatures] = self.graph.getVertex(vertexIndices[i])
            testX[:, numFeatures:numFeatures*2] = self.graph.getVertexList().getVertices(allVertexIndices)

            testX = self.preprocessor.process(testX)
            _, scores = self.learningAlg.classify(testX)
            P[i, :], S[i, :] = self.indicesFromScores(vertexIndices[i], scores)

        return P, S

