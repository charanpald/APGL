
import logging
import time
import numpy 
 
from apgl.egograph.EgoNetworkSimulator import EgoNetworkSimulator
from apgl.util import * 
from apgl.io import *
from apgl.graph import * 
from apgl.predictors.edge import *
from apgl.predictors import *
from apgl.kernel import * 

"""
This class does parameter selection for a general learner in an ego simulation and then
runs the experiments and saves results.
"""

class InfoExperiment:
    @staticmethod
    def trainPredictor():
        """
        Train the model for the simulation.
        """
        paramsFile = InfoExperiment.getSvmParamsFileName()
        sampleSize = InfoExperiment.getNumSimulationExamples()

        simulator = EgoNetworkSimulator(examplesFileName)
        params, paramFuncs = InfoExperiment.loadParams(paramsFile)

        startTime = time.time()
        simulator.trainClassifier(params, paramFuncs, sampleSize)
        endTime = time.time()
        logging.info("Time taken for training is " + str((endTime-startTime)) + " seconds.")

        return simulator

    @staticmethod
    def getNumVertices():
        return 10000

    @staticmethod
    def getNumCVExamples():
        return 15000

    @staticmethod
    def getNumSimulationExamples():
        return 20000

    @staticmethod 
    def getParamsFileName():
        return PathDefaults.getOutputDir() + "diffusion/" + "learnerParams.pkl"

    @staticmethod
    def getGraphFileName():
        return PathDefaults.getDataDir() + "infoDiffusion/" + "EgoAlterDecays.dat"

    @staticmethod
    def loadParams(paramsFile, predictor):
        paramsList = predictor.loadParams(paramsFile)
        return paramsList 

    @staticmethod
    def saveParams(paramsFile):
        """
        This method runs model selection for the SVM and saves the parameters
        and errors. Only need to do this once for the data.
        """
        graphFileName = InfoExperiment.getGraphFileName()
        graph  = SparseGraph.load(graphFileName)

        logging.info("Find all ego networks")
        trees = graph.findTrees()
        subgraphSize = 10000
        subgraphIndices = []
        

        for i in range(len(trees)):
            subgraphIndices.extend(trees[i])
            
            if len(subgraphIndices) > subgraphSize:
                logging.info("Chose " + str(i) + " ego networks.")
                break 

        graph = graph.subgraph(subgraphIndices)
        logging.info("Taking random subgraph of size " + str(graph.getNumVertices()))

        folds = 3
        sampleSize = graph.getNumEdges()

        lambda1s = 2.0**numpy.arange(-8,-2)
        lambda2s = 2.0**numpy.arange(-8,-2)
        sigmas = 2.0**numpy.arange(-6,0)

        #lambda1s = [0.0625, 2.0, 10.0]
        #lambda2s = [0.0625, 5.0, 30.0]
        #sigmas = [0.0625, 1.0, 10.0]

        logging.info("lambda1s = " + str(lambda1s))
        logging.info("lambda2s = " + str(lambda2s))
        logging.info("sigmas = " + str(sigmas))

        lmbda = 0.1
        alpha = 5.0 #Note that this is not the same as that used the errorFunc
        kernel = LinearKernel()
        alterRegressor = PrimalWeightedRidgeRegression(lmbda, alpha)
        egoRegressor = KernelRidgeRegression(kernel, lmbda)
        predictor = EgoEdgeLabelPredictor(alterRegressor, egoRegressor)

        simulator = EgoNetworkSimulator(graph, predictor)
        errorFunc = Evaluator.weightedRootMeanSqError

        paramList = []
        paramFuncs = [egoRegressor.setLambda, alterRegressor.setLambda]

        #First just use the linear kernel 
        for i in lambda1s:
            for j in lambda2s:
                paramList.append([i, j])

        params, paramFuncs, error = simulator.modelSelection(paramList, paramFuncs, folds, errorFunc, sampleSize)
        
        #Now try the RBF kernel
        kernel = GaussianKernel()
        egoRegressor.setKernel(kernel)

        paramFuncs2 = [egoRegressor.setLambda, alterRegressor.setLambda, kernel.setSigma]
        paramList2 = []

        for i in lambda1s:
            for j in lambda2s:
                for k in sigmas:
                    paramList2.append([i, j, k])

        params2, paramFuncs2, error2 = simulator.modelSelection(paramList2, paramFuncs2, folds, errorFunc, sampleSize)

        if error2 < error:
            params = params2
            paramFuncs = paramFuncs2

        paramsFile = InfoExperiment.getParamsFileName()
        (means, vars) = simulator.evaluateClassifier(params, paramFuncs, folds, errorFunc, sampleSize)

        logging.info("Evaluated classifier with mean errors " + str(means))
        simulator.getClassifier().saveParams(params, paramFuncs, paramsFile)
