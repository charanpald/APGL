import logging

from apgl.data.Standardiser import Standardiser
from apgl.egograph import *
from apgl.graph import  *
from apgl.util import * 
import numpy

class EgoNetworkSimulator(AbstractDiffusionSimulator):
    """
    A class which combines Ego network prediction with simulating information transmission
    within a simulated social network.
    """
    def __init__(self, graph, predictor):
        """
        Create the class by reading a graph with labelled edges. Instantiate the predictor
        and create a preprocesor to standarise examples to have zero mean and unit variance.
        """
        self.graph = graph
        self.predictor = predictor
        self.errorMethod = Evaluator.balancedError

        #Note: We modify the vertices of the input graph!!!!
        logging.warn("About to modify (normalise) the vertices of the graph.")
        self.preprocessor = Standardiser()
        V = graph.getVertexList().getVertices(graph.getAllVertexIds())
        V = self.preprocessor.normaliseArray(V)
        graph.getVertexList().setVertices(V)

    def getPreprocessor(self):
        """
        Returns the preprocessor
        """
        return self.preprocessor

    def sampleEdges(self, sampleSize):
        """
        This function exists so that we can sample the same examples used in model
        selection and exclude them when running evaluateClassifier.
        """
        edges = self.graph.getAllEdges()
        trainInds = numpy.random.permutation(edges.shape[0])[0:sampleSize]
        trainEdges = edges[trainInds, :]

        trainGraph = SparseGraph(self.graph.getVertexList(), self.graph.isUndirected())
        trainGraph.addEdges(trainEdges, self.graph.getEdgeValues(trainEdges))

        logging.info("Randomly sampled " + str(sampleSize) + " edges")

        return trainGraph

    def modelSelection(self, paramList, paramFunc, folds, errorFunc, sampleSize):
        """
        Perform model selection using an edge label predictor. 
        """
        Parameter.checkInt(folds, 0, sampleSize)
        Parameter.checkInt(sampleSize, 0, self.graph.getNumEdges()) 

        #trainGraph = self.sampleEdges(sampleSize)
        trainGraph = self.graph

        #Perform model selection
        meanErrs, stdErrs = self.predictor.cvModelSelection(trainGraph, paramList, paramFunc, folds, errorFunc)
        logging.info("Model selection errors:" + str(meanErrs))
        logging.info("Model selection stds:" + str(stdErrs))
        logging.info("Model selection best parameters:" + str(paramList[numpy.argmin(meanErrs)]))

        return paramList[numpy.argmin(meanErrs)], paramFunc, meanErrs[numpy.argmin(meanErrs)] 

    def evaluateClassifier(self, params, paramFuncs, folds, errorFunc, sampleSize, invert=True):
        """
        Evaluate the predictor with the given parameters. Often model selection is done before this step
        and in that case, invert=True uses a sample excluding those used for model selection.

        Return a set of errors for each
        """
        Parameter.checkInt(folds, 0, sampleSize)
        Parameter.checkInt(sampleSize, 0, self.graph.getNumEdges())

        trainGraph = self.sampleEdges(sampleSize)

        return self.predictor.cvError(trainGraph, params, paramFuncs, folds, errorFunc)

    def trainClassifier(self, params, paramFuncs, sampleSize):
        
        for j in range(len(params)):
            paramFuncs[j](params[j])

        trainGraph = self.sampleEdges(sampleSize)
        self.predictor.learnModel(trainGraph)

        return self.predictor

    def runSimulation(self, maxIterations):
        Parameter.checkInt(maxIterations, 1, float('inf'))

        #Notice that the data is preprocessed in the same way as the survey data
        egoSimulator = EgoSimulator(self.graph, self.predictor, self.preprocessor)

        totalInfo = numpy.zeros(maxIterations+1)
        totalInfo[0] = EgoUtils.getTotalInformation(self.graph)
        logging.info("Total number of people with information: " + str(totalInfo[0]))

        logging.info("--- Simulation Started ---")

        for i in range(0, maxIterations):
            logging.info("--- Iteration " + str(i) + " ---")

            self.graph = egoSimulator.advanceGraph()
            totalInfo[i+1] = EgoUtils.getTotalInformation(self.graph)
            logging.info("Total number of people with information: " + str(totalInfo[i+1]))

            #Compute distribution of ages etc. in alters
            alterIndices = egoSimulator.getAlters(i)
            alterAges = numpy.zeros(len(alterIndices))
            alterGenders = numpy.zeros(len(alterIndices))

            for j in range(0, len(alterIndices)):
                currentVertex = self.graph.getVertex(alterIndices[j])
                alterAges[j] = currentVertex[self.egoQuestionIds.index(("Q5X", 0))]
                alterGenders[j] = currentVertex[self.egoQuestionIds.index(("Q4", 0))]

            (freqs, items) = Util.histogram(alterAges)
            logging.info("Distribution of ages " + str(freqs) + " " + str(items))
            (freqs, items) = Util.histogram(alterGenders)
            logging.info("Distribution of genders " + str(freqs) + " " + str(items))

        logging.info("--- Simulation Finished ---")

        return totalInfo, egoSimulator.getTransmissions()

    def getVertexFeatureDistribution(self, fIndex, vIndices=None):
        return self.graph.getVertexFeatureDistribution(fIndex, vIndices)

    def getPreProcessor(self):
        return self.preprocessor

    def getClassifier(self):
        return self.predictor

    preprocessor = None
    examplesList = None
    predictor = None
    graph = None
    edgeWeight = 1