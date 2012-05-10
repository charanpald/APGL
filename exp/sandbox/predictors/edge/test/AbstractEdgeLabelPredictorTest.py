

import unittest
import logging
from exp.sandbox.predictors.edge.EgoEdgeLabelPredictor import EgoEdgeLabelPredictor
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Evaluator import Evaluator
from apgl.graph.VertexList import VertexList
from apgl.graph.SparseGraph import SparseGraph
from apgl.predictors import * 
import numpy 


class  AbstractEdgeLabelPredictorTest(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(21)
        numVertices = 10
        numFeatures = 5

        vList = VertexList(numVertices, numFeatures)
        vList.setVertices(numpy.random.rand(numVertices, numFeatures))
        graph = SparseGraph(vList, False)

        graph.addEdge(0, 1, 1)
        graph.addEdge(0, 2, 1)
        graph.addEdge(0, 3, 1)
        graph.addEdge(0, 4, -1)
        graph.addEdge(0, 5, -1)
        graph.addEdge(1, 2, 1)
        graph.addEdge(1, 3, -1)
        graph.addEdge(1, 8, 1)
        graph.addEdge(2, 3, -1)
        graph.addEdge(2, 4, -1)
        graph.addEdge(2, 5, -1)
        graph.addEdge(2, 6, 1)

        self.graph = graph 

    def testCvModelSelection(self):
 
        graph = self.graph

        lmbda = 0.01

        alterRegressor = PrimalRidgeRegression(lmbda)
        egoRegressor = PrimalRidgeRegression(lmbda)
        predictor = EgoEdgeLabelPredictor(alterRegressor, egoRegressor)

        paramList = [[0.1, 0.1], [0.2, 0.1], [0.1, 0.2]]
        paramFunc = [egoRegressor.setLambda, alterRegressor.setLambda]
        folds = 3
        errorFunc = Evaluator.rootMeanSqError

        meanErs, stdEr =  predictor.cvModelSelection(graph, paramList, paramFunc, folds, errorFunc)

        logging.debug(meanErs)
        self.assertTrue(meanErs.shape[0] == len(paramList))
        meanErs2 = meanErs

        paramList = [[0.1, 0.1], [0.2, 0.1]]
        meanErs, stdEr =  predictor.cvModelSelection(graph, paramList, paramFunc, folds, errorFunc)
        logging.debug(meanErs)
        self.assertTrue(meanErs.shape[0] == len(paramList))
        self.assertTrue((meanErs2[0:2] == meanErs).all())


        paramList = [[0.1, 0.1]]
        meanErs, stdEr =  predictor.cvModelSelection(graph, paramList, paramFunc, folds, errorFunc)
        logging.debug(meanErs)
        self.assertTrue(meanErs.shape[0] == len(paramList))
        self.assertTrue((meanErs2[0:1] == meanErs).all())

    def testCvError(self):
        graph = self.graph
        lmbda = 0.01

        alterRegressor = PrimalRidgeRegression(lmbda)
        egoRegressor = PrimalRidgeRegression(lmbda)
        predictor = EgoEdgeLabelPredictor(alterRegressor, egoRegressor)

        paramList = [0.1, 0.1]
        paramFunc = [egoRegressor.setLambda, alterRegressor.setLambda]
        folds = 3
        errorFunc = Evaluator.rootMeanSqError

        meanEr, stdEr =  predictor.cvError(graph, paramList, paramFunc, folds, errorFunc)

        paramList = [[0.1, 0.1]]
        meanErs, stdEr =  predictor.cvModelSelection(graph, paramList, paramFunc, folds, errorFunc)

        self.assertEquals(meanEr, meanErs[0])

    def testSaveParams(self):
        try:
            lmbda = 0.01

            alterRegressor = PrimalRidgeRegression(lmbda)
            egoRegressor = PrimalRidgeRegression(lmbda)
            predictor = EgoEdgeLabelPredictor(alterRegressor, egoRegressor)

            params = [0.1, 0.2]
            paramFuncs = [egoRegressor.setLambda, alterRegressor.setLambda]
            fileName = PathDefaults.getTempDir() + "tempParams.pkl"

            predictor.saveParams(params, paramFuncs, fileName)
        except IOError as e:
            logging.warn(e)

    def testLoadParams(self):
        try:
            lmbda = 0.01

            alterRegressor = PrimalRidgeRegression(lmbda)
            egoRegressor = PrimalRidgeRegression(lmbda)
            predictor = EgoEdgeLabelPredictor(alterRegressor, egoRegressor)

            params = [0.1, 0.2]
            paramFuncs = [egoRegressor.setLambda, alterRegressor.setLambda]
            fileName = PathDefaults.getTempDir() + "tempParams.pkl"

            predictor.saveParams(params, paramFuncs, fileName)

            params2 = predictor.loadParams(fileName)

            self.assertTrue(params2[0][0] == "apgl.predictors.PrimalRidgeRegression")
            self.assertTrue(params2[0][1] == "setLambda")
            self.assertTrue(params2[0][2] == 0.1)

            self.assertTrue(params2[1][0] == "apgl.predictors.PrimalRidgeRegression")
            self.assertTrue(params2[1][1] == "setLambda")
            self.assertTrue(params2[1][2] == 0.2)
        except IOError as e:
            logging.warn(e)

if __name__ == '__main__':
    unittest.main()

