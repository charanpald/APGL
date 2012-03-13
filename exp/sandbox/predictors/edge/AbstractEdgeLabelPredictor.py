
import numpy
import logging
import pickle 
from apgl.util.Sampling import Sampling
from apgl.graph.SparseGraph import SparseGraph

class AbstractEdgeLabelPredictor(object):

    def learnModel(self, graph):
        """
        Take a graph with labelled edges and learn the function which maps from
        a pair of vertices to the corresponding value of the edge.
        """
        pass


    def predictEdges(self, graph, edges):
        """
        Take as input a graph a list of possible edges and predict the value of 
        the label of the edge. 
        """
        pass

    def cvModelSelection(self, graph, paramList, paramFunc, folds, errorFunc):
        """
        ParamList is a list of lists of parameters and paramFunc
        is a list of the corresponding functions to call with the parameters
        as arguments. Note that a parameter can also be a tuple which is expanded
        out before the function is called. 

        e.g.
        paramList = [[1, 2], [2, 1], [12, 1]]
        paramFunc = [predictor.setC, predictor.setD]
        """

        inds = Sampling.crossValidation(folds, graph.getNumEdges())
        errors = numpy.zeros((len(paramList), folds))
        allEdges = graph.getAllEdges()

        for i in range(len(paramList)):
            paramSet = paramList[i]
            logging.debug("Using paramSet=" + str(paramSet))

            for j in range(len(paramSet)):
                if type(paramSet[j]) == tuple:
                    paramFunc[j](*paramSet[j])
                else: 
                    paramFunc[j](paramSet[j])

            predY = numpy.zeros(0)
            y = numpy.zeros(0)
            j = 0 

            for (trainInds, testInds) in inds:
                trainEdges = allEdges[trainInds, :]
                testEdges = allEdges[testInds, :]

                trainGraph = SparseGraph(graph.getVertexList(), graph.isUndirected())
                trainGraph.addEdges(trainEdges, graph.getEdgeValues(trainEdges))

                testGraph = SparseGraph(graph.getVertexList(), graph.isUndirected())
                testGraph.addEdges(testEdges, graph.getEdgeValues(testEdges))

                self.learnModel(trainGraph)

                predY = self.predictEdges(testGraph, testGraph.getAllEdges())
                y = testGraph.getEdgeValues(testGraph.getAllEdges())
                #Note that the order the edges is different in testGraphs as
                #opposed to graph when calling getAllEdges()

                errors[i, j] = errorFunc(y, predY)
                j = j+1 

            logging.info("Error of current fold: " + str(numpy.mean(errors[i, :])))

        meanErrors = numpy.mean(errors, 1)
        strErrors = numpy.std(errors, 1)

        return meanErrors, strErrors

    def cvError(self, graph, params, paramFuncs, folds, errorFunc):
        """
        Compute the cross validation error over the graph with a given set of
        parameters. 
        """

        paramList =[params]
        meanError, strError = self.cvModelSelection(graph, paramList, paramFuncs, folds, errorFunc)

        return meanError, strError


    def saveParams(self, params, paramFuncs, fileName):
        """
        Write out the  parameters to a file using a list of lists with each parameter
        represented by the dict in indexed by <module str> and with value [<function str>, <value>]
        """

        logging.info("Writing parameters in fixed order")
        file = open(fileName, "wb")
        paramsList = []

        for i in range(len(params)):
            paramList = [paramFuncs[i].__module__, paramFuncs[i].__name__, params[i]]
            paramsList.append(paramList)

        pickle.dump(paramsList, file)
        logging.info("Saved parameters as file " + fileName )

    def loadParams(self, fileName):

        file = open(fileName, "rb")
        paramsList = pickle.load(file)
        logging.info("Loaded parameters " + str(len(paramsList)) + " from file " + fileName)
        return paramsList