


import unittest
import logging
import sys
import numpy
from apgl.egograph.EgoNetworkSimulator import EgoNetworkSimulator
from apgl.util import * 
from apgl.graph import *
from apgl.generator import *
from apgl.predictors import *
from apgl.predictors.edge import *

class EgoNetworkSimulatorTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        dataDir = PathDefaults.getDataDir() + "infoDiffusion/"

        numVertices = 100
        numFeatures = 5

        c = numpy.random.rand(numFeatures)

        vList = VertexList(numVertices, numFeatures)
        vList.setVertices(numpy.random.rand(numVertices, numFeatures))
        graph = SparseGraph(vList)

        p = 0.1
        generator = ErdosRenyiGenerator(p)
        graph = generator.generate(graph)

        #Now rewrite some vertices to have different labels which depend only
        #on the alter. 
        edges = graph.getAllEdges()

        for i in range(edges.shape[0]):
            edgeLabel = numpy.dot(graph.getVertex(edges[i, 1]), c)
            graph.addEdge(edges[i, 0], edges[i, 1], edgeLabel)

        #Create the predictor
        lmbda = 0.01

        self.alterRegressor = PrimalRidgeRegression(lmbda)
        self.egoRegressor = PrimalRidgeRegression(lmbda)
        predictor = EgoEdgeLabelPredictor(self.alterRegressor, self.egoRegressor)

        self.egoNetworkSimulator = EgoNetworkSimulator(graph, predictor)
        self.graph = graph 

    def testSampleExamples(self):
        numEdges = 100 
        graph = self.egoNetworkSimulator.sampleEdges(numEdges)

        self.assertEquals(graph.getNumEdges(), numEdges)
        self.assertEquals(graph.getNumVertices(), self.graph.getNumVertices())
        self.assertEquals(graph.isUndirected(), self.graph.isUndirected())


    def testModelSelection(self):
        paramList = [[0.1, 0.1], [0.2, 0.1], [0.1, 0.2]]
        paramFunc = [self.egoRegressor.setLambda, self.alterRegressor.setLambda] 
        folds = 3
        errorFunc = Evaluator.rootMeanSqError
        sampleSize = 100 

        params, paramFuncs, errors = self.egoNetworkSimulator.modelSelection(paramList, paramFunc, folds, errorFunc, sampleSize)

        logging.debug(params)
        logging.debug(paramFuncs)
        logging.debug(errors)

    def testEvaluateClassifier(self):
        params = [0.1, 0.1]
        paramFunc = [self.egoRegressor.setLambda, self.alterRegressor.setLambda]
        folds = 3
        errorFunc = Evaluator.rootMeanSqError
        sampleSize = 100 

        (means, vars) = self.egoNetworkSimulator.evaluateClassifier(params, paramFunc, folds, errorFunc, sampleSize)

        logging.debug(means)
        logging.debug(vars)

    def testTrainClassifier(self):
        params= [0.1, 0.1]
        paramFuncs = [self.egoRegressor.setLambda, self.alterRegressor.setLambda]
        sampleSize = 100 

        self.egoNetworkSimulator.trainClassifier(params, paramFuncs, sampleSize)

    def testGenerateRandomGraph(self):
        egoFileName = PathDefaults.getDataDir() + "infoDiffusion/EgoData.csv"
        alterFileName = PathDefaults.getDataDir()  + "infoDiffusion/AlterData.csv"
        numVertices = 1000
        infoProb = 0.1
        graphType = "SmallWorld"
        p = 0.1
        neighbours = 10

        self.egoNetworkSimulator.generateRandomGraph(egoFileName, alterFileName, numVertices, infoProb, graphType, p, neighbours)

    def testRunSimulation(self):
        egoFileName = PathDefaults.getDataDir() + "infoDiffusion/EgoData.csv"
        alterFileName = PathDefaults.getDataDir()  + "infoDiffusion/AlterData.csv"
        numVertices = 1000
        infoProb = 0.1
        graphType = "SmallWorld"
        p = 0.1
        neighbours = 10

        params= [0.1, 0.1]
        paramFuncs = [self.egoRegressor.setLambda, self.alterRegressor.setLambda]
        sampleSize = 100 

        maxIterations = 5

        self.egoNetworkSimulator.trainClassifier(params, paramFuncs, sampleSize)
        self.egoNetworkSimulator.generateRandomGraph(egoFileName, alterFileName, numVertices, infoProb, graphType, p, neighbours)
        self.egoNetworkSimulator.runSimulation(maxIterations)


if __name__ == '__main__':
    unittest.main()

