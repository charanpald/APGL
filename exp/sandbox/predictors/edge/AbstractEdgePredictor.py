
import numpy
import logging 
from apgl.util import *
from apgl.graph.SparseGraph import SparseGraph


class AbstractEdgePredictor(object):
    def __init__(self):
        Util.abstract()

    def setWindowSize(self, windowSize):
        self.windowSize = windowSize

    def cvModelSelection(self, graph, paramList, paramFunc, folds):
        """
        ParamList is a list of lists of parameters and paramFunc
        is a list of the corresponding functions to call with the parameters
        as arguments.

        e.g. 
        paramList = [[1, 2], [2, 1], [12, 1]]
        paramFunc = [predictor.setC, predictor.setD]
        """

        inds = Sampling.crossValidation(folds, graph.getNumEdges())
        errors = numpy.zeros(len(paramList))
        allEdges = graph.getAllEdges()
        numTestExamples = 0 

        for i in range(len(paramList)):
            paramSet = paramList[i]
            logging.debug("Using paramSet=" + str(paramSet))

            for j in range(len(paramSet)):
                paramFunc[j](paramSet[j])

            for (trainInds, testInds) in inds:
                trainEdges = allEdges[trainInds, :]
                testEdges = allEdges[testInds, :]

                trainGraph = SparseGraph(graph.getVertexList())
                trainGraph.addEdges(trainEdges)

                self.learnModel(trainGraph)
                P, S = self.predictEdges(testEdges[:, 0])

                for k in range(testEdges.shape[0]):
                    if not testEdges[k, 1] in P[k,:]:
                        errors[i] += 1.0

                numTestExamples += testEdges.shape[0]

        return errors/numTestExamples
                        

    def indicesFromScores(self, index, scores):
        neighbours = self.graph.neighbours(index)
        scores[neighbours] = -float('Inf')
        logging.debug(scores)

        indices = numpy.flipud(numpy.argsort(scores))
        indices = indices[0: self.windowSize]

        return indices, scores[indices]