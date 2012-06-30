'''
Created on 24 Jul 2009

@author: charanpal
'''

import unittest
import logging
import numpy
import apgl
import sys 
from apgl.predictors.LibSVM import LibSVM, computeTestError, computePenalisedError, computePenalty, computeIdealPenalty, computeBootstrapError
from apgl.data.ExamplesGenerator import ExamplesGenerator 
from apgl.util.Evaluator import Evaluator
from apgl.util.Sampling import Sampling
from apgl.data.Standardiser import Standardiser


@apgl.skipIf(not apgl.checkImport('sklearn'), 'Module svm is required')
class LibSVMTest(unittest.TestCase):
    def setUp(self):
        try:
            import sklearn
        except ImportError as error:
            logging.debug(error)
            return 

        numpy.random.seed(21)
        numExamples = 100
        numFeatures = 10
        eg = ExamplesGenerator()

        self.X, self.y = eg.generateBinaryExamples(numExamples, numFeatures)
        self.svm = LibSVM()
        self.svm.Cs = 2.0**numpy.arange(-2, 2, dtype=numpy.float)
        self.svm.gammas = 2.0**numpy.arange(-3, 1, dtype=numpy.float)
        self.svm.epsilons = 2.0**numpy.arange(-2, 0, dtype=numpy.float)

        numpy.set_printoptions(linewidth=150, suppress=True, precision=3)
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    def testLearnModel(self):
        try:
            import sklearn
        except ImportError as error:
            return

        self.svm.learnModel(self.X, self.y)
        predY = self.svm.classify(self.X)
        y = self.y

        e = Evaluator.binaryError(y, predY)

        #Test for wrong labels
        numExamples = 6
        X = numpy.array([[-3], [-2], [-1], [1], [2] ,[3]], numpy.float)
        y = numpy.array([[-1], [-1], [-1], [1], [1] ,[5]])

        self.assertRaises(ValueError, self.svm.learnModel, X, y)

        #Try the regression SVM
        svm = LibSVM(type="Epsilon_SVR")
        y = numpy.random.rand(self.X.shape[0])
        svm.learnModel(self.X, self.y)
        

    def testSetErrorCost(self):
        try:
            import sklearn
        except ImportError as error:
            return

        numExamples = 1000
        numFeatures = 100
        eg = ExamplesGenerator()
        X, y = eg.generateBinaryExamples(numExamples, numFeatures)
        svm = LibSVM()

        C = 0.1
        kernel = "linear"
        kernelParam = 0
        svm.setKernel(kernel, kernelParam)
        svm.setC(C)

        svm.setErrorCost(0.1)
        svm.learnModel(X, y)
        predY = svm.classify(X)
        e1 = Evaluator.binaryErrorP(y, predY)

        svm.setErrorCost(0.9)
        svm.learnModel(X, y)
        predY = svm.classify(X)
        e2 = Evaluator.binaryErrorP(y, predY)

        self.assertTrue(e1 > e2)

    def testClassify(self):
        try:
            import sklearn
        except ImportError as error:
            return

        self.svm.learnModel(self.X, self.y)
        predY = self.svm.classify(self.X)
        y = self.y

        e = Evaluator.binaryError(y, predY)

        #Now, permute examples
        perm = numpy.random.permutation(self.X.shape[0])
        predY = self.svm.classify(self.X[perm, :])
        y = y[perm]

        e2 = Evaluator.binaryError(y, predY)

        self.assertEquals(e, e2)

    def testEvaluateCv(self):
        try:
            import sklearn
        except ImportError as error:
            return

        folds = 10
        (means, vars) = self.svm.evaluateCv(self.X, self.y, folds)

        self.assertTrue((means <= 1).all())
        self.assertTrue((means>= 0).all())
        self.assertTrue((vars <= 1).all())
        self.assertTrue((vars>= 0).all())

    @apgl.skip("")
    def testGetModel(self):
        try:
            import sklearn
        except ImportError as error:
            return

        numExamples = 50
        numFeatures = 3
        eg = ExamplesGenerator()

        X, y = eg.generateBinaryExamples(numExamples, numFeatures)
        svm = LibSVM()
        svm.learnModel(X, y)

        weights, b  = svm.getWeights()

        #logging.debug(weights)
        #logging.debug(b)
        

    @apgl.skip("")
    def testGetWeights(self):
        try:
            import sklearn
        except ImportError as error:
            return

        numExamples = 6
        X = numpy.array([[-3], [-2], [-1], [1], [2] ,[3]], numpy.float64)
        #X = numpy.random.rand(numExamples, 10)
        y = numpy.array([[-1], [-1], [-1], [1], [1] ,[1]])

        svm = LibSVM()
        svm.learnModel(X, y.ravel())
        weights, b  = svm.getWeights()

        #Let's see if we can compute the decision values 
        y, decisions = svm.predict(X, True)
        decisions2 = numpy.zeros(numExamples)
        decisions2 = numpy.dot(X, weights) - b

        self.assertTrue((decisions == decisions2).all())
        predY = numpy.sign(decisions2)
        self.assertTrue((y.ravel() == predY).all())

        #Do the same test on a random datasets
        numExamples = 50
        numFeatures = 10

        X = numpy.random.rand(numExamples, numFeatures)
        y = numpy.sign(numpy.random.rand(numExamples)-0.5)

        svm = LibSVM()
        svm.learnModel(X, y.ravel())
        weights, b  = svm.getWeights()

        #Let's see if we can compute the decision values
        y, decisions = svm.predict(X, True)
        decisions2 = numpy.dot(X, weights) + b

        tol = 10**-6

        self.assertTrue(numpy.linalg.norm(decisions - decisions2) < tol)
        predY = numpy.sign(decisions2)
        self.assertTrue((y.ravel() == predY).all())

    def testSetTermination(self):
        try:
            import sklearn
        except ImportError as error:
            return


        self.svm.learnModel(self.X, self.y)
        self.svm.setTermination(0.1)
        self.svm.learnModel(self.X, self.y)

    def testSetSvmType(self):
        try:
            import sklearn
        except ImportError as error:
            return

        numExamples = 100
        numFeatures = 10
        X = numpy.random.randn(numExamples, numFeatures)
        X = Standardiser().standardiseArray(X)
        c = numpy.random.randn(numFeatures)

        y = numpy.dot(X, numpy.array([c]).T).ravel() + 1
        y2 = numpy.array(y > 0, numpy.int32)*2 -1 
        
        svm = LibSVM()

        svm.setSvmType("Epsilon_SVR")

        self.assertEquals(svm.getType(), "Epsilon_SVR")

        #Try to get a good error
        Cs = 2**numpy.arange(-6, 4, dtype=numpy.float)
        epsilons = 2**numpy.arange(-6, 4, dtype=numpy.float)

        bestError = 10 
        for C in Cs:
            for epsilon in epsilons:
                svm.setEpsilon(epsilon)
                svm.setC(C)
                svm.learnModel(X, y)
                yp = svm.predict(X)

                if Evaluator.rootMeanSqError(y, yp) < bestError:
                    bestError = Evaluator.rootMeanSqError(y, yp) 

        self.assertTrue(bestError < Evaluator.rootMeanSqError(y, numpy.zeros(y.shape[0])))
        
        svm.setSvmType("C_SVC")
        svm.learnModel(X, y2)
        yp2 = svm.predict(X)

        self.assertTrue(0 <= Evaluator.binaryError(y2, yp2)  <= 1)

    @apgl.skip("")
    def testSaveParams(self):
        try:
            import sklearn
        except ImportError as error:
            return

        svm = LibSVM()
        svm.setC(10.5)
        svm.setEpsilon(12.1)
        svm.setErrorCost(1.8)
        svm.setSvmType("Epsilon_SVR")
        svm.setTermination(0.12)
        svm.setKernel("gaussian", 0.43)

        outputDir = PathDefaults.getOutputDir()
        fileName = outputDir + "test/testSvmParams"
        svm.saveParams(fileName)

        svm2 = LibSVM()
        svm2.loadParams(fileName)

        self.assertEquals(svm.getC(), 10.5)
        self.assertEquals(svm.getEpsilon(), 12.1)
        self.assertEqual(svm.getErrorCost(), 1.8)
        self.assertEqual(svm.getSvmType(), "Epsilon_SVR")
        self.assertEqual(svm.getTermination(), 0.12)
        self.assertEqual(svm.getKernel(), "gaussian")
        self.assertEqual(svm.getKernelParams(), 0.43)

    def testStr(self):
        try:
            import sklearn
        except ImportError as error:
            return

        svm = LibSVM()

        #logging.debug(svm)

    def testSetEpsilon(self):
        """
        Test out the parameter for the regressive SVM, vary epsilon and look at
        number of support vectors. 
        """
        try:
            import sklearn
        except ImportError as error:
            return

        svm = LibSVM()
        svm.setC(10.0)
        svm.setEpsilon(0.1)
        svm.setSvmType("Epsilon_SVR")

        numExamples = 100
        numFeatures = 10
        X = numpy.random.randn(numExamples, numFeatures)
        c = numpy.random.randn(numFeatures)

        y = numpy.dot(X, numpy.array([c]).T).ravel() + numpy.random.randn(100)
        
        svm.setEpsilon(1.0)
        svm.learnModel(X, y)
        numSV = svm.getModel().support_.shape
        
        svm.setEpsilon(0.5)
        svm.learnModel(X, y)
        numSV2 = svm.getModel().support_.shape

        svm.setEpsilon(0.01)
        svm.learnModel(X, y)
        numSV3 = svm.getModel().support_.shape

        #There should be fewer SVs as epsilon increases
        self.assertTrue(numSV < numSV2)
        self.assertTrue(numSV2 < numSV3)

    def testPredict(self):
        try:
            import sklearn
        except ImportError as error:
            return

        numExamples = 100
        numFeatures = 10
        X = numpy.random.randn(numExamples, numFeatures)
        c = numpy.random.randn(numFeatures)

        y = numpy.dot(X, numpy.array([c]).T).ravel()
        y = numpy.array(y > 0, numpy.int32)*2 -1

        svm = LibSVM()
        svm.learnModel(X, y)
        y2, d = svm.predict(X, True)

        #self.assertTrue((numpy.sign(d) == y2).all())

    #@unittest.skip("")
    def testParallelVfcvRbf(self):
        folds = 3
        idx = Sampling.crossValidation(folds, self.X.shape[0])
        svm = self.svm
        svm.setKernel("gaussian")
        bestSVM, meanErrors = svm.parallelVfcvRbf(self.X, self.y, idx)

        tol = 10**-6 
        bestError = 1
        meanErrors2 = numpy.zeros((svm.Cs.shape[0], svm.gammas.shape[0])) 

        for i in range(svm.Cs.shape[0]):
            C = svm.Cs[i]
            for j in range(svm.gammas.shape[0]):
                gamma = svm.gammas[j]
                error = 0
                for trainInds, testInds in idx:
                    trainX = self.X[trainInds, :]
                    trainY = self.y[trainInds]
                    testX = self.X[testInds, :]
                    testY = self.y[testInds]

                    svm.setGamma(gamma)
                    svm.setC(C)
                    svm.learnModel(trainX, trainY)
                    predY = svm.predict(testX)
                    error += Evaluator.binaryError(predY, testY)

                meanErrors2[i, j] = error/len(idx)

                if error < bestError:
                    bestC = C
                    bestGamma = gamma
                    bestError = error

        self.assertEquals(bestC, bestSVM.getC())
        self.assertEquals(bestGamma, bestSVM.getGamma())
        self.assertTrue(numpy.linalg.norm(meanErrors2.T - meanErrors) < tol)

    def testParallelVfcvRbf2(self):
        #In this test we try SVM regression 
        folds = 3
        idx = Sampling.crossValidation(folds, self.X.shape[0])
        svm = self.svm
        svm.setKernel("gaussian")
        svm.setSvmType("Epsilon_SVR")
        bestSVM, meanErrors = svm.parallelVfcvRbf(self.X, self.y, idx, type="Epsilon_SVR")

        tol = 10**-6
        bestError = 100
        meanErrors2 = numpy.zeros((svm.gammas.shape[0], svm.epsilons.shape[0], svm.Cs.shape[0]))

        for i in range(svm.Cs.shape[0]):
            C = svm.Cs[i]
            for j in range(svm.gammas.shape[0]):
                gamma = svm.gammas[j]
                for k in range(svm.epsilons.shape[0]):
                    epsilon = svm.epsilons[k]

                    error = 0
                    for trainInds, testInds in idx:
                        trainX = self.X[trainInds, :]
                        trainY = self.y[trainInds]
                        testX = self.X[testInds, :]
                        testY = self.y[testInds]

                        svm.setGamma(gamma)
                        svm.setC(C)
                        svm.setEpsilon(epsilon)
                        svm.learnModel(trainX, trainY)
                        predY = svm.predict(testX)
                        error += svm.getMetricMethod()(predY, testY)

                    meanErrors2[j, k, i] = error/len(idx)

                    if error < bestError:
                        bestC = C
                        bestGamma = gamma
                        bestError = error
                        bestEpsilon = epsilon

        self.assertEquals(bestC, bestSVM.getC())
        self.assertEquals(bestGamma, bestSVM.getGamma())
        self.assertEquals(bestEpsilon, bestSVM.getEpsilon())
        self.assertTrue(numpy.linalg.norm(meanErrors2 - meanErrors) < tol)

    def testParallelVfPenRbf(self):
        folds = 3
        Cv = numpy.array([4.0])
        idx = Sampling.crossValidation(folds, self.X.shape[0])
        svm = self.svm
        svm.setKernel("gaussian")
        resultsList = svm.parallelVfPenRbf(self.X, self.y, idx, Cv)

        tol = 10**-6
        bestError = 1
        meanErrors2 = numpy.zeros((svm.Cs.shape[0], svm.gammas.shape[0]))

        for i in range(svm.Cs.shape[0]):
            C = svm.Cs[i]
            for j in range(svm.gammas.shape[0]):
                gamma = svm.gammas[j]
                penalty = 0
                for trainInds, testInds in idx:
                    trainX = self.X[trainInds, :]
                    trainY = self.y[trainInds]

                    svm.setGamma(gamma)
                    svm.setC(C)
                    svm.learnModel(trainX, trainY)
                    predY = svm.predict(self.X)
                    predTrainY = svm.predict(trainX)
                    penalty += Evaluator.binaryError(predY, self.y) - Evaluator.binaryError(predTrainY, trainY)

                penalty = penalty*Cv[0]/len(idx)
                svm.learnModel(self.X, self.y)
                predY = svm.predict(self.X)
                meanErrors2[i, j] = Evaluator.binaryError(predY, self.y) + penalty

                if meanErrors2[i, j] < bestError:
                    bestC = C
                    bestGamma = gamma
                    bestError = meanErrors2[i, j]

        bestSVM, trainErrors, currentPenalties = resultsList[0]
        meanErrors = trainErrors + currentPenalties

        self.assertEquals(bestC, bestSVM.getC())
        self.assertEquals(bestGamma, bestSVM.getGamma())
        self.assertTrue(numpy.linalg.norm(meanErrors2.T - meanErrors) < tol)

    #@unittest.skip("")
    def testParallelVfPenRbf2(self):
        #Test support vector regression 
        folds = 3
        Cv = numpy.array([4.0])
        idx = Sampling.crossValidation(folds, self.X.shape[0])
        svm = self.svm
        svm.setKernel("gaussian")
        svm.setSvmType("Epsilon_SVR")
        resultsList = svm.parallelVfPenRbf(self.X, self.y, idx, Cv, type="Epsilon_SVR")

        tol = 10**-6 
        bestError = 100
        meanErrors2 = numpy.zeros((svm.gammas.shape[0], svm.epsilons.shape[0], svm.Cs.shape[0]))

        for i in range(svm.Cs.shape[0]):
            C = svm.Cs[i]
            for j in range(svm.gammas.shape[0]):
                gamma = svm.gammas[j]
                for k in range(svm.epsilons.shape[0]):
                    epsilon = svm.epsilons[k]
                    
                    penalty = 0
                    for trainInds, testInds in idx:
                        trainX = self.X[trainInds, :]
                        trainY = self.y[trainInds]

                        svm.setGamma(gamma)
                        svm.setC(C)
                        svm.setEpsilon(epsilon)
                        svm.learnModel(trainX, trainY)
                        predY = svm.predict(self.X)
                        predTrainY = svm.predict(trainX)
                        penalty += svm.getMetricMethod()(predY, self.y) - svm.getMetricMethod()(predTrainY, trainY)

                    penalty = penalty*Cv[0]/len(idx)
                    svm.learnModel(self.X, self.y)
                    predY = svm.predict(self.X)
                    meanErrors2[j, k, i] = svm.getMetricMethod()(predY, self.y) + penalty

                    if meanErrors2[j, k, i] < bestError:
                        bestC = C
                        bestGamma = gamma
                        bestEpsilon = epsilon 
                        bestError = meanErrors2[j, k, i]

        bestSVM, trainErrors, currentPenalties = resultsList[0]
        meanErrors = trainErrors + currentPenalties

        self.assertEquals(bestC, bestSVM.getC())
        self.assertEquals(bestGamma, bestSVM.getGamma())
        self.assertEquals(bestEpsilon, bestSVM.getEpsilon())
        self.assertTrue(numpy.linalg.norm(meanErrors2 - meanErrors) < tol)


    def testGetC(self):
        svm = LibSVM()
        svm.setC(10.0)
        C = svm.getC()
        self.assertTrue(C == 10.0)

    def testGetGamma(self):
        svm = LibSVM()
        svm.setKernel("gaussian", 12.0)
        gamma = svm.getKernelParams()
        self.assertTrue(gamma == 12.0)

    def testComputeTestError(self):
        C = 10.0
        gamma = 0.5

        numTrainExamples = self.X.shape[0]*0.5

        trainX, trainY = self.X[0:numTrainExamples, :], self.y[0:numTrainExamples]
        testX, testY = self.X[numTrainExamples:, :], self.y[numTrainExamples:]

        svm = LibSVM('gaussian', gamma, C)
        args = (trainX, trainY, testX, testY, svm)
        error = computeTestError(args)

        svm = LibSVM('gaussian', gamma, C)
        svm.learnModel(trainX, trainY)
        predY = svm.predict(testX)
        self.assertEquals(Evaluator.binaryError(predY, testY), error)

    def testComputeBootstrapError(self):
        C = 10.0
        gamma = 0.5

        numTrainExamples = self.X.shape[0]*0.5

        trainX, trainY = self.X[0:numTrainExamples, :], self.y[0:numTrainExamples]
        testX, testY = self.X[numTrainExamples:, :], self.y[numTrainExamples:]
        
        svm = LibSVM('gaussian', gamma, C)

        args = (trainX, trainY, testX, testY, svm)
        error = computeBootstrapError(args)


    def testComputePenalisedError(self):
        C = 10.0
        gamma = 0.5
        Cv = 4
        folds = 5

        idx = Sampling.crossValidation(folds, self.y.shape[0])
        svm = LibSVM('gaussian', gamma, C)

        args = (self.X, self.y, idx, svm, Cv)
        error = computePenalisedError(args)

    def testComputePenalty(self):
        C = 10.0
        gamma = 0.5
        Cv = 4
        folds = 5

        idx = Sampling.crossValidation(folds, self.y.shape[0])
        svm = LibSVM("gaussian", gamma, C)

        args = (self.X, self.y, idx, svm, Cv)
        error = computePenalty(args)

    def testComputeIdealPenalty(self):
        C = 10.0
        gamma = 0.5
        svm = LibSVM("gaussian", gamma, C)
        args = (self.X, self.y, self.X, self.y, svm)
        error = computeIdealPenalty(args)

    def testParallelPenaltyGridRbf(self):
        svm = self.svm
        svm.setKernel("gaussian")
        trainX = self.X[0:40, :]
        trainY = self.y[0:40]

        idealPenalties = svm.parallelPenaltyGridRbf(trainX, trainY, self.X, self.y)
        idealPenalties2 = numpy.zeros((svm.Cs.shape[0], svm.gammas.shape[0]))
        idealPenalties3 = numpy.zeros((svm.Cs.shape[0], svm.gammas.shape[0]))

        for i in range(svm.Cs.shape[0]):
            C = svm.Cs[i]
            for j in range(svm.gammas.shape[0]):
                gamma = svm.gammas[j]

                svm.setGamma(gamma)
                svm.setC(C)
                svm.learnModel(trainX, trainY)
                predY = svm.predict(self.X)
                predTrainY = svm.predict(trainX)
                penalty = Evaluator.binaryError(predY, self.y) - Evaluator.binaryError(predTrainY, trainY)

                idealPenalties2[i, j] = penalty

                args = (trainX, trainY, self.X, self.y, svm)
                idealPenalties3[i, j] = computeIdealPenalty(args)

        tol = 10**-6 
        self.assertTrue(numpy.linalg.norm(idealPenalties2.T - idealPenalties) < tol)


    def testParallelPenaltyGridRbf2(self):
        #Test with SVM regression
        svm = self.svm
        svm.setKernel("gaussian")
        svm.setSvmType("Epsilon_SVR")
        trainX = self.X[0:40, :]
        trainY = self.y[0:40]

        idealPenalties = svm.parallelPenaltyGridRbf(trainX, trainY, self.X, self.y, type="Epsilon_SVR")
        idealPenalties2 = numpy.zeros((svm.gammas.shape[0], svm.epsilons.shape[0], svm.Cs.shape[0]))

        for i in range(svm.Cs.shape[0]):
            C = svm.Cs[i]
            for j in range(svm.gammas.shape[0]):
                gamma = svm.gammas[j]
                for k in range(svm.epsilons.shape[0]):
                    epsilon = svm.epsilons[k]
                    svm.setGamma(gamma)
                    svm.setC(C)
                    svm.setEpsilon(epsilon)

                    svm.learnModel(trainX, trainY)
                    predY = svm.predict(self.X)
                    predTrainY = svm.predict(trainX)
                    penalty = svm.getMetricMethod()(predY, self.y) - svm.getMetricMethod()(predTrainY, trainY)

                    idealPenalties2[j, k, i] = penalty

        tol = 10**-6
        self.assertTrue(numpy.linalg.norm(idealPenalties2 - idealPenalties) < tol)


    def testParallelModelSelect(self):
        folds = 3
        idx = Sampling.crossValidation(folds, self.X.shape[0])
        svm = self.svm
        svm.setKernel("gaussian")

        paramDict = {} 
        paramDict["setC"] = svm.getCs()
        paramDict["setGamma"] = svm.getGammas()    
        
        bestSVM, meanErrors = svm.parallelModelSelect(self.X, self.y, idx, paramDict)
        
        tol = 10**-6 
        bestError = 1
        meanErrors2 = numpy.zeros((svm.Cs.shape[0], svm.gammas.shape[0])) 

        for i in range(svm.Cs.shape[0]):
            C = svm.Cs[i]
            for j in range(svm.gammas.shape[0]):
                gamma = svm.gammas[j]
                error = 0
                for trainInds, testInds in idx:
                    trainX = self.X[trainInds, :]
                    trainY = self.y[trainInds]
                    testX = self.X[testInds, :]
                    testY = self.y[testInds]

                    svm.setGamma(gamma)
                    svm.setC(C)
                    svm.learnModel(trainX, trainY)
                    predY = svm.predict(testX)
                    error += Evaluator.binaryError(predY, testY)

                meanErrors2[i, j] = error/len(idx)

                if error < bestError:
                    bestC = C
                    bestGamma = gamma
                    bestError = error
                    
        self.assertEquals(bestC, bestSVM.getC())
        self.assertEquals(bestGamma, bestSVM.getGamma())
        self.assertTrue(numpy.linalg.norm(meanErrors2.T - meanErrors) < tol)


    def testParallelPenaltyGrid2(self):
        #Test with SVM regression
        svm = self.svm
        svm.setKernel("gaussian")
        svm.setSvmType("Epsilon_SVR")
        trainX = self.X[0:40, :]
        trainY = self.y[0:40]
        
        paramDict = {} 
        paramDict["setC"] = svm.getCs()
        paramDict["setGamma"] = svm.getGammas()  
        paramDict["setEpsilon"] = svm.getEpsilons()
        
        #print(paramDict.keys())

        idealPenalties = svm.parallelPenaltyGrid(trainX, trainY, self.X, self.y, paramDict)
        idealPenalties2 = numpy.zeros((svm.gammas.shape[0], svm.epsilons.shape[0], svm.Cs.shape[0]))

        for i in range(svm.Cs.shape[0]):
            C = svm.Cs[i]
            for j in range(svm.gammas.shape[0]):
                gamma = svm.gammas[j]
                for k in range(svm.epsilons.shape[0]):
                    epsilon = svm.epsilons[k]
                    svm.setGamma(gamma)
                    svm.setC(C)
                    svm.setEpsilon(epsilon)

                    svm.learnModel(trainX, trainY)
                    predY = svm.predict(self.X)
                    predTrainY = svm.predict(trainX)
                    penalty = svm.getMetricMethod()(predY, self.y) - svm.getMetricMethod()(predTrainY, trainY)

                    idealPenalties2[j, k, i] = penalty

        tol = 10**-6
        self.assertTrue(numpy.linalg.norm(idealPenalties2 - idealPenalties) < tol)

    def testParallelPen(self): 
        folds = 3
        Cv = numpy.array([4.0])
        idx = Sampling.crossValidation(folds, self.X.shape[0])
        svm = self.svm
        svm.setKernel("gaussian")

        paramDict = {} 
        paramDict["setC"] = svm.getCs()
        paramDict["setGamma"] = svm.getGammas()            
        
        resultsList = svm.parallelPen(self.X, self.y, idx, paramDict, Cv)
        
        tol = 10**-6
        bestError = 1
        trainErrors2 = numpy.zeros((svm.Cs.shape[0], svm.gammas.shape[0]))
        penalties2 = numpy.zeros((svm.Cs.shape[0], svm.gammas.shape[0]))
        meanErrors2 = numpy.zeros((svm.Cs.shape[0], svm.gammas.shape[0]))

        for i in range(svm.Cs.shape[0]):
            C = svm.Cs[i]
            for j in range(svm.gammas.shape[0]):
                gamma = svm.gammas[j]
                penalty = 0
                for trainInds, testInds in idx:
                    trainX = self.X[trainInds, :]
                    trainY = self.y[trainInds]

                    svm.setGamma(gamma)
                    svm.setC(C)
                    svm.learnModel(trainX, trainY)
                    predY = svm.predict(self.X)
                    predTrainY = svm.predict(trainX)
                    penalty += Evaluator.binaryError(predY, self.y) - Evaluator.binaryError(predTrainY, trainY)

                penalty = penalty*Cv[0]/len(idx)
                svm.learnModel(self.X, self.y)
                predY = svm.predict(self.X)
                trainErrors2[i, j] = Evaluator.binaryError(predY, self.y)
                penalties2[i, j] = penalty
                meanErrors2[i, j] = Evaluator.binaryError(predY, self.y) + penalty

                if meanErrors2[i, j] < bestError:
                    bestC = C
                    bestGamma = gamma
                    bestError = meanErrors2[i, j]

        bestSVM, trainErrors, currentPenalties = resultsList[0]
        meanErrors = trainErrors + currentPenalties

        self.assertEquals(bestC, bestSVM.getC())
        self.assertEquals(bestGamma, bestSVM.getGamma())
        self.assertTrue(numpy.linalg.norm(meanErrors2.T - meanErrors) < tol)
        self.assertTrue(numpy.linalg.norm(trainErrors2.T - trainErrors) < tol)
        self.assertTrue(numpy.linalg.norm(penalties2.T - currentPenalties) < tol)

    def testParallelPenaltyGrid(self):
        svm = self.svm
        svm.setKernel("gaussian")
        trainX = self.X[0:40, :]
        trainY = self.y[0:40]
        
        paramDict = {} 
        paramDict["setC"] = svm.getCs()
        paramDict["setGamma"] = svm.getGammas()      

        idealPenalties = svm.parallelPenaltyGrid(trainX, trainY, self.X, self.y, paramDict)
        idealPenalties2 = numpy.zeros((svm.Cs.shape[0], svm.gammas.shape[0]))
        idealPenalties3 = numpy.zeros((svm.Cs.shape[0], svm.gammas.shape[0]))

        for i in range(svm.Cs.shape[0]):
            C = svm.Cs[i]
            for j in range(svm.gammas.shape[0]):
                gamma = svm.gammas[j]

                svm.setGamma(gamma)
                svm.setC(C)
                svm.learnModel(trainX, trainY)
                predY = svm.predict(self.X)
                predTrainY = svm.predict(trainX)
                penalty = Evaluator.binaryError(predY, self.y) - Evaluator.binaryError(predTrainY, trainY)

                idealPenalties2[i, j] = penalty

                args = (trainX, trainY, self.X, self.y, svm)
                idealPenalties3[i, j] = computeIdealPenalty(args)

        tol = 10**-6 
        self.assertTrue(numpy.linalg.norm(idealPenalties2.T - idealPenalties) < tol)

        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
