
import svm
import numpy
import unittest
import logging
import numpy 
from apgl.modelselect.ModelSelectUtils import ModelSelectUtils, computeIdealPenalty
from apgl.predictors.LibSVM import LibSVM, computeIdealPenalty as computeIdealPenalty2
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Evaluator import Evaluator
import matplotlib.pyplot as plt 

class  MetabolomicsUtilsTestCase(unittest.TestCase):

    @unittest.skip("")
    def testBayesError(self):
        dataDir = PathDefaults.getDataDir() + "modelPenalisation/toy/"
        data = numpy.load(dataDir + "toyData.npz")
        gridPoints, X, y, pdfX, pdfY1X, pdfYminus1X = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"], data["arr_4"], data["arr_5"]

        sampleSize = 100
        trainX, trainY = X[0:sampleSize, :], y[0:sampleSize]
        testX, testY = X[sampleSize:, :], y[sampleSize:]

        #We form a test set from the grid points
        gridX = numpy.zeros((gridPoints.shape[0]**2, 2))
        for m in range(gridPoints.shape[0]):
            gridX[m*gridPoints.shape[0]:(m+1)*gridPoints.shape[0], 0] = gridPoints
            gridX[m*gridPoints.shape[0]:(m+1)*gridPoints.shape[0], 1] = gridPoints[m]

        Cs = 2**numpy.arange(-5, 5, dtype=numpy.float)
        gammas = 2**numpy.arange(-5, 5, dtype=numpy.float)

        bestError = 1 

        for C in Cs:
            for gamma in gammas:
                svm = LibSVM(kernel="gaussian", C=C, kernelParam=gamma)
                svm.learnModel(trainX, trainY)
                predY, decisionsY = svm.predict(gridX, True)
                decisionGrid = numpy.reshape(decisionsY, (gridPoints.shape[0], gridPoints.shape[0]), order="F")
                error = ModelSelectUtils.bayesError(gridPoints, decisionGrid, pdfX, pdfY1X, pdfYminus1X)

                predY, decisionsY = svm.predict(testX, True)
                error2 = Evaluator.binaryError(testY, predY)
                print(error, error2)

                if error < bestError:
                    error = bestError
                    bestC = C
                    bestGamma = gamma

        svm = LibSVM(kernel="gaussian", C=bestC, kernelParam=bestGamma)
        svm.learnModel(trainX, trainY)
        predY, decisionsY = svm.predict(gridX, True)

        plt.figure(0)
        plt.contourf(gridPoints, gridPoints, decisionGrid, 100)
        plt.colorbar()

        plt.figure(1)
        plt.scatter(X[y==1, 0], X[y==1, 1], c='r' ,label="-1")
        plt.scatter(X[y==-1, 0], X[y==-1, 1], c='b',label="+1")
        plt.legend()
        plt.show()

    def testComputeIdealPenalty(self):
        dataDir = PathDefaults.getDataDir() + "modelPenalisation/toy/"
        data = numpy.load(dataDir + "toyData.npz")
        gridPoints, X, y, pdfX, pdfY1X, pdfYminus1X = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"], data["arr_4"], data["arr_5"]

        sampleSize = 100
        trainX, trainY = X[0:sampleSize, :], y[0:sampleSize]
        testX, testY = X[sampleSize:, :], y[sampleSize:]

        #We form a test set from the grid points
        fullX = numpy.zeros((gridPoints.shape[0]**2, 2))
        for m in range(gridPoints.shape[0]):
            fullX[m*gridPoints.shape[0]:(m+1)*gridPoints.shape[0], 0] = gridPoints
            fullX[m*gridPoints.shape[0]:(m+1)*gridPoints.shape[0], 1] = gridPoints[m]

        C = 1.0
        gamma = 1.0
        args = (trainX, trainY, fullX, C, gamma, gridPoints, pdfX, pdfY1X, pdfYminus1X)
        penalty = computeIdealPenalty(args)


        #Now compute penalty using data
        args = (trainX, trainY, testX, testY, C, gamma)
        penalty2 = computeIdealPenalty2(args)

        self.assertAlmostEquals(penalty2, penalty, 2)
        

if __name__ == '__main__':
    unittest.main()

