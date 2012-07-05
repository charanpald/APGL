'''
Created on 16 Jul 2009

@author: charanpal
'''
from apgl.util.Util import Util
from apgl.util.Evaluator import Evaluator 
from apgl.util.Parameter import Parameter
from apgl.util.Sampling import Sampling
import numpy
import logging
import gc
import itertools 
import multiprocessing 


#Start with some functions used for multiprocessing 

def computeTestError(args):
    """
    Used in conjunction with the parallel model selection. Trains and then tests
    on a seperate test set. 
    """
    (trainX, trainY, testX, testY, learner) = args
    learner.learnModel(trainX, trainY)
    predY = learner.predict(testX)

    return learner.getMetricMethod()(testY, predY)

def computeBootstrapError(args):
    """
    Used in conjunction with the parallel model selection. Trains and then tests
    on a seperate test set and evaluated the bootstrap error. 
    """
    (trainX, trainY, testX, testY, learner) = args
    learner.learnModel(trainX, trainY)
    predTestY = learner.predict(testX)
    predTrainY = learner.predict(trainX)
    weight = 0.632
    return Evaluator.binaryBootstrapError(predTestY, testY, predTrainY, trainY, weight)

def computePenalisedError(args):
    """
    Used in conjunction with the parallel model selection. It returns the
    binary error on the whole training set and the penalty
    """
    (X, y, idx, learner, Cv) = args
    penalty = computePenalty(args)
    learner.learnModel(X, y)
    predY = learner.predict(X)
    return learner.getMetricMethod()(predY, y), penalty

def computePenalty(args):
    """
    Used in conjunction with the parallel model selection.
    """
    (X, y, idx, learner, Cv) = args
    penalty = 0
    folds = len(idx)

    for idxtr, idxts in idx:
        trainX, trainY = X[idxtr, :], y[idxtr]
        learner.learnModel(trainX, trainY)
        predY = learner.predict(X)
        predTrainY = learner.predict(trainX)
        penalty += learner.getMetricMethod()(predY, y) - learner.getMetricMethod()(predTrainY, trainY)
            
    penalty *= Cv/folds
    return penalty

def computeIdealPenalty(args):
    """
    Find the complete penalty.
    """
    (trainX, trainY, fullX, fullY, learner) = args

    learner.learnModel(trainX, trainY)
    predTrainY = learner.predict(trainX)
    predFullY = learner.predict(fullX)

    idealPenalty = learner.getMetricMethod()(predFullY, fullY) - learner.getMetricMethod()(predTrainY, trainY)
    return idealPenalty

class AbstractPredictor(object):
    def __init__(self): 
        #Used in multiprocessing code 
        self.processes = multiprocessing.cpu_count() 
        self.chunkSize = 10 
    """
    An abstract classifier for binary labelled data. 
    """
    def learnModel(self, X, y):
        """
        Learn a model for a set of examples given as the rows of the matrix X,
        with corresponding labels given in the elements of 1D array y.

        :param X: A matrix with examples as rows
        :type X: :class:`ndarray`

        :param y: A vector of labels
        :type y: :class:`ndarray`
        """
        Util.abstract()
    
    def predict(self, X):
        """
        Make a prediction for a set of examples given as the rows of the matrix X.

        :param X: A matrix with examples as rows
        :type X: :class:`ndarray`
        """
        Util.abstract()

    def evaluateStratifiedCv(self, X, y, folds, metricMethod=Evaluator.binaryError):
        """
        Compute the stratified cross validation according to a given metric.
        """
        try:
            from scikits.learn import cross_val
            Parameter.checkInt(folds, 2, float('inf'))
            idx = cross_val.StratifiedKFold(y, folds)
            metrics = AbstractPredictor.evaluateLearn(X, y, idx, self.learnModel, self.predict, metricMethod)

            mean = numpy.mean(metrics, 0)
            var = numpy.var(metrics, 0)

            return (mean, var)

        except ImportError:
            logging.warn("Failed to import scikits")
            raise 


    def evaluateCv(self, X, y, folds, metricMethod=Evaluator.binaryError):
        """
        Compute the cross validation according to a given metric. 
        """
        Parameter.checkInt(folds, 2, float('inf'))
        idx = Sampling.crossValidation(folds, y.shape[0])
        metrics = AbstractPredictor.evaluateLearn(X, y, idx, self.learnModel, self.predict, metricMethod)

        mean = numpy.mean(metrics, 0)
        var = numpy.var(metrics, 0)

        return (mean, var)

    @staticmethod
    def evaluateLearn(X, y, idx, learnModel, predict, metricMethod, progress=True):
        """
        Evaluate this learning algorithm using the given list of training/test splits 
        The metricMethod is a method which takes (predictedY, realY) as input
        and returns a metric about the quality of the evaluation.

        :param X: A matrix with examples as rows 
        :type X: :class:`ndarray`

        :param y: A vector of labels 
        :type y: :class:`ndarray`

        :param idx: A list of training/test splits 
        :type idx: :class:`list`

        :param learnModel: A function such that learnModel(X, y) finds a mapping from X to y 
        :type learnModel: :class:`function`

        :param predict: A function such that predict(X) makes predictions for X
        :type predict: :class:`function`

        :param metricMethod: A function such that metricMethod(predY, testY) returns the quality of predicted labels predY
        :type metricMethod: :class:`function`

        Output: the mean and variation of the cross validation folds. 
        """
        #Parameter.checkClass(idx, list)
        Parameter.checkClass(X, numpy.ndarray)
        Parameter.checkArray(X, softCheck=True)
        Parameter.checkInt(X.shape[0], 1, float('inf'))
        Parameter.checkClass(y, numpy.ndarray)
        Parameter.checkArray(y, softCheck=True)

        if y.ndim != 1:
            raise ValueError("Dimention of y must be 1")
        
        i = 0
        metrics = numpy.zeros(len(idx))
        logging.debug("EvaluateLearn: Using " + str(len(idx)) + " splits on " + str(X.shape[0]) + " examples")

        for idxtr, idxts in idx:
            if progress:
                Util.printConciseIteration(i, 1, len(idx))

            trainX, testX = X[idxtr, :], X[idxts, :]
            trainY, testY = y[idxtr], y[idxts]
            #logging.debug("Distribution of labels in evaluateLearn train: " + str(numpy.bincount(trainY)))
            #logging.debug("Distribution of labels in evaluateLearn test: " + str(numpy.bincount(testY)))

            learnModel(trainX, trainY)
            predY = predict(testX)
            gc.collect()

            metrics[i] = metricMethod(predY, testY)
            i += 1

        return metrics

    @staticmethod
    def evaluateLearners(X, Y, indexList, splitFunction, learnerIterator, metricMethods, progress=True):
        """
        Perform model selection and output an average metric over a number of train/test
        splits as defined by idx. Finds the *minimum* model according to the evaluation
        of the predicted labels with metricMethods[0]. The variable metricMethods is a list
        of functions to call metricMethod(predY, trueY) of which the first is used
        in model selection.
        """
        Parameter.checkClass(X, numpy.ndarray)
        Parameter.checkArray(X, softCheck=True)
        Parameter.checkClass(Y, numpy.ndarray)
        Parameter.checkArray(Y, softCheck=True)

        if Y.ndim != 1:
            raise ValueError("Expecting Y to be 1D")

        i = 0
        mainMetricMethod = metricMethods[0]

        bestLearners = []
        allMetrics = []

        for trainInds, testInds in indexList:
            trainX = X[trainInds, :]
            trainY = Y[trainInds]

            testX = X[testInds, :]
            testY = Y[testInds]

            minMetric = float('inf')

            for learner in learnerIterator:
                logging.debug("Learning with " + str(learner))
                idx = splitFunction(trainX, trainY)
                metrics = AbstractPredictor.evaluateLearn(trainX, trainY, idx, learner.learnModel, learner.predict, mainMetricMethod, progress)

                meanMetric = numpy.mean(metrics)
                stdMetric = numpy.std(metrics)

                if meanMetric < minMetric:
                    bestLearner = learner
                    minMetric = meanMetric

                #Try to get some memory back
                gc.collect()

            bestLearner.learnModel(trainX, trainY)
            predY = bestLearner.predict(testX)

            bestLearners.append(bestLearner)

            #Now compute all metrics
            currentMetrics = []
            for metricMethod in metricMethods:
                currentMetrics.append(metricMethod(predY, testY))

            allMetrics.append(currentMetrics)
            logging.debug("Outer metric(s): " + str(currentMetrics))
            i += 1

        for i in range(len(allMetrics)):
            logging.debug("Learner = " + str(bestLearners[i]) + " error= " + str(allMetrics[i]))
        logging.debug("All done")

        return allMetrics, bestLearners

    @staticmethod 
    def evaluateLearn2(X, Y, indexList, learnModel, predict, metricMethods):
        """
        Evaluate a learner given  functions (learnModel, predict)
        and save metrics on the training and test sets given by metric methods.

        #Could combine this with evaluateLearn 
        """
        Parameter.checkClass(X, numpy.ndarray)
        Parameter.checkArray(X, softCheck=True)
        Parameter.checkClass(Y, numpy.ndarray)
        Parameter.checkArray(Y, softCheck=True)

        if Y.ndim != 1:
            raise ValueError("Expecting Y to be 1D")

        trainMetrics = []
        testMetrics = []
        for i in range(len(metricMethods)): 
            trainMetrics.append([])
            testMetrics.append([])

        for trainInds, testInds in indexList:
            trainX, trainY = X[trainInds, :], Y[trainInds]
            testX, testY = X[testInds, :], Y[testInds]

            learnModel(trainX, trainY)
            predTrainY = predict(trainX)
            predTestY = predict(testX)

            #Now compute all metrics
            i = 0 
            for metricMethod in metricMethods:
                trainMetrics[i].append(metricMethod(trainY, predTrainY))
                testMetrics[i].append(metricMethod(testY, predTestY))
                i += 1 

            gc.collect()

        logging.debug("All done")

        return trainMetrics, testMetrics

    def parallelModelSelect(self, X, y, idx, paramDict):
        """
        Perform parallel model selection using any learner. 
        Using the best set of parameters train using the whole dataset.

        :param X: The examples as rows
        :type X: :class:`numpy.ndarray`

        :param y: The binary -1/+1 labels 
        :type y: :class:`numpy.ndarray`

        :param idx: A list of train/test splits
        
        :param paramDict: A dictionary index by the method name and with value as an array of values
        :type X: :class:`dict`
        """
        Parameter.checkClass(X, numpy.ndarray)
        Parameter.checkClass(y, numpy.ndarray)
        folds = len(idx)

        gridSize = [] 
        gridInds = [] 
        for key in paramDict.keys(): 
            gridSize.append(paramDict[key].shape[0])
            gridInds.append(numpy.arange(paramDict[key].shape[0])) 
            
        meanErrors = numpy.zeros(tuple(gridSize))
        m = 0

        for trainInds, testInds in idx:
            Util.printConciseIteration(m, 1, folds, "CV Inner fold: ")
            trainX, trainY = X[trainInds, :], y[trainInds]
            testX, testY = X[testInds, :], y[testInds]
            
            indexIter = itertools.product(*gridInds)
            paramList = []
            for inds in indexIter: 
                learner = self.copy()     
                currentInd = 0             
            
                for key, val in paramDict.items():
                    method = getattr(learner, key)
                    method(val[inds[currentInd]])
                    currentInd += 1                    
                
                paramList.append((trainX, trainY, testX, testY, learner))
            
            pool = multiprocessing.Pool(processes=self.processes, maxtasksperchild=100)
            resultsIterator = pool.imap(computeTestError, paramList, self.chunkSize)
            #resultsIterator =  itertools.imap(computeTestError, paramList)
        
            indexIter = itertools.product(*gridInds)
            for inds in indexIter: 
                meanErrors[inds] += resultsIterator.next()/float(folds)

            pool.terminate()

            m += 1 

        bestInds = numpy.unravel_index(numpy.argmin(meanErrors), meanErrors.shape)
        currentInd = 0    
        learner = self.copy()         
    
        for key, val in paramDict.items():
            method = getattr(learner, key)
            method(val[bestInds[currentInd]])
            currentInd += 1   
        
        learner.learnModel(X, y)

        return learner, meanErrors

    def parallelPen(self, X, y, idx, paramDict, Cvs):
        """
        Perform parallel penalisation using any learner. 
        Using the best set of parameters train using the whole dataset.

        :param X: The examples as rows
        :type X: :class:`numpy.ndarray`

        :param y: The binary -1/+1 labels 
        :type y: :class:`numpy.ndarray`

        :param idx: A list of train/test splits

        :param paramDict: A dictionary index by the method name and with value as an array of values
        :type X: :class:`dict`

        """
        Parameter.checkClass(X, numpy.ndarray)
        Parameter.checkClass(y, numpy.ndarray)
        folds = len(idx)

        gridSize = [] 
        gridInds = [] 
        for key in paramDict.keys(): 
            gridSize.append(paramDict[key].shape[0])
            gridInds.append(numpy.arange(paramDict[key].shape[0])) 
            
        trainErrors = numpy.zeros(tuple(gridSize))
        penalties = numpy.zeros(tuple(gridSize))

        indexIter = itertools.product(*gridInds)
        paramList = []
        for inds in indexIter: 
            learner = self.copy()     
            currentInd = 0             
        
            for key, val in paramDict.items():
                method = getattr(learner, key)
                method(val[inds[currentInd]])
                currentInd += 1                    
            
            paramList.append((X, y, idx, learner, 1.0))

        pool = multiprocessing.Pool(processes=self.processes, maxtasksperchild=100)
        resultsIterator = pool.imap(computePenalisedError, paramList, self.chunkSize)

        indexIter = itertools.product(*gridInds)
        for inds in indexIter: 
            trainErrors[inds], penalties[inds] = resultsIterator.next()

        pool.terminate()

        #Store v fold penalised error
        #In the case that Cv < 0 we use the corrected penalisation 
        resultsList = []
        for k in range(Cvs.shape[0]):
            Cv = Cvs[k]
            
            if Cv >= 0: 
                logging.debug("Computing penalisation of Cv=" + str(Cv))
                currentPenalties = penalties*Cv
            else: 
                logging.debug("Computing corrected penalisation with sigma=" + str(abs(Cv)))
                sigma = abs(Cv)
                dynamicCv = (folds-1)*(1-numpy.exp(-sigma*trainErrors)) + float(folds)*numpy.exp(-sigma*trainErrors)    
                currentPenalties = penalties*dynamicCv
                
            meanErrors = trainErrors + currentPenalties
            
            bestInds = numpy.unravel_index(numpy.argmin(meanErrors), meanErrors.shape)
            currentInd = 0    
            learner = self.copy()         
        
            for key, val in paramDict.items():
                method = getattr(learner, key)
                method(val[bestInds[currentInd]])
                currentInd += 1   
            
            learner.learnModel(X, y)            
            resultsList.append((learner, trainErrors, currentPenalties))

        return resultsList
    
    def gridShape(self, paramDict): 
        gridSize = [] 

        for key in paramDict.keys(): 
            gridSize.append(paramDict[key].shape[0])
            
        return tuple(gridSize)
            
    
    def parallelPenaltyGrid(self, trainX, trainY, fullX, fullY, paramDict, errorFunc=computeIdealPenalty):
        """
        Find out the "ideal" penalty using a training set and the full dataset. If one specifies 
        a different error function then that is computed over the grid of parameters. 
        """
        Parameter.checkClass(trainX, numpy.ndarray)
        Parameter.checkClass(trainY, numpy.ndarray)

        gridSize = [] 
        gridInds = [] 
        for key in paramDict.keys(): 
            gridSize.append(paramDict[key].shape[0])
            gridInds.append(numpy.arange(paramDict[key].shape[0])) 
            
        idealPenalties = numpy.zeros(tuple(gridSize))

        indexIter = itertools.product(*gridInds)
        paramList = []
        for inds in indexIter: 
            learner = self.copy()     
            currentInd = 0             
        
            for key, val in paramDict.items():
                method = getattr(learner, key)
                method(val[inds[currentInd]])
                currentInd += 1                    
            
            paramList.append((trainX, trainY, fullX, fullY, learner))
        
        pool = multiprocessing.Pool(processes=self.processes, maxtasksperchild=100)
        resultsIterator = pool.imap(errorFunc, paramList, self.chunkSize)
        indexIter = itertools.product(*gridInds)
        
        for inds in indexIter: 
            idealPenalties[inds] = resultsIterator.next()

        pool.terminate()

        return idealPenalties

    def parallelSplitGrid(self, trainX, trainY, testX, testY, paramDict):
        """
        Find out the "ideal" error using a training set and the full dataset. 
        """
        return self.parallelPenaltyGrid(trainX, trainY, testX, testY, paramDict, computeTestError)
   
    def getParamsArray(self, paramDict): 
        """
        A method to return an array of parameters for a given paramDict 
        """
        paramsArray = numpy.ndarray(len(paramDict))
        currentInd = 0 
        
        for key, val in paramDict.items():
            key = key.replace("s", "g", 1)
            
            method = getattr(self, key)
            paramsArray[currentInd] = method()
            currentInd += 1 
            
        return paramsArray 
        
    def setChunkSize(self, chunkSize): 
        Parameter.checkInt(chunkSize, 1, float("inf"))
        self.chunkSize = chunkSize 