import numpy
import multiprocessing 

from apgl.util.Parameter import Parameter
from apgl.util.Evaluator import Evaluator
from apgl.predictors.LibSVM import LibSVM

def computeIdealPenalty(args):
    """
    Find the complete penalty.
    """
    (X, y, fullX, C, gamma, gridPoints, pdfX, pdfY1X, pdfYminus1X) = args

    svm = LibSVM('gaussian', gamma, C)
    svm.learnModel(X, y)
    predY = svm.predict(X)
    predFullY, decisionsY = svm.predict(fullX, True)
    decisionGrid = numpy.reshape(decisionsY, (gridPoints.shape[0], gridPoints.shape[0]), order="F")
    trueError = ModelSelectUtils.bayesError(gridPoints, decisionGrid, pdfX, pdfY1X, pdfYminus1X)
    idealPenalty = trueError - Evaluator.binaryError(predY, y)

    return idealPenalty

def parallelPenaltyGridRbf(svm, X, y, fullX, gridPoints, pdfX, pdfY1X, pdfYminus1X):
    """
    Find out the "ideal" penalty.
    """
    Parameter.checkClass(X, numpy.ndarray)
    Parameter.checkClass(y, numpy.ndarray)
    chunkSize = 10

    idealPenalties = numpy.zeros((svm.Cs.shape[0], svm.gammas.shape[0]))
    paramList = []

    for i in range(svm.Cs.shape[0]):
        for j in range(svm.gammas.shape[0]):
            paramList.append((X, y, fullX, svm.Cs[i], svm.gammas[j], gridPoints, pdfX, pdfY1X, pdfYminus1X))

    pool = multiprocessing.Pool()
    resultsIterator = pool.imap(computeIdealPenalty, paramList, chunkSize)

    for i in range(svm.Cs.shape[0]):
        for j in range(svm.gammas.shape[0]):
            idealPenalties[i, j] = resultsIterator.next()

    pool.terminate()

    return idealPenalties

class ModelSelectUtils(object):
    @staticmethod
    def loadRatschDataset(dataDir, datasetName, realisationIndex):
        trainDatasetName = dataDir + datasetName + "/" + datasetName + "_train_data_" + str(realisationIndex + 1) + ".asc"
        trainLabelsName = dataDir + datasetName + "/" + datasetName + "_train_labels_" + str(realisationIndex + 1) + ".asc"
        testDatasetName = dataDir + datasetName + "/" + datasetName + "_test_data_" + str(realisationIndex + 1) + ".asc"
        testLabelsName = dataDir + datasetName + "/" + datasetName + "_test_labels_" + str(realisationIndex + 1) + ".asc"

        trainX = numpy.loadtxt(trainDatasetName, delimiter=None)
        trainY = numpy.loadtxt(trainLabelsName, delimiter=None)
        testX = numpy.loadtxt(testDatasetName, delimiter=None)
        testY = numpy.loadtxt(testLabelsName, delimiter=None)
        
        trainY = numpy.array(trainY, numpy.int)
        testY = numpy.array(testY, numpy.int)

        return trainX, trainY, testX, testY 

    @staticmethod
    def loadRegressDataset(dataDir, datasetName, realisationIndex):
        examplesName = dataDir + datasetName + "/data.txt"
        XY = numpy.loadtxt(examplesName, delimiter=",")

        trainIndsName = dataDir + datasetName + "/trainInds.txt"
        trainInds = numpy.loadtxt(trainIndsName, delimiter=",", dtype=numpy.int)

        testIndsName = dataDir  + datasetName + "/testInds.txt"
        testInds = numpy.loadtxt(testIndsName, delimiter=",", dtype=numpy.int)

        trainX = XY[trainInds[realisationIndex, :], :-1]
        trainY = XY[trainInds[realisationIndex, :], -1]
        testX = XY[testInds[realisationIndex, :], :-1]
        testY = XY[testInds[realisationIndex, :], -1]

        return trainX, trainY, testX, testY

    @staticmethod
    def getRegressionDatasets(withRealisations=False):
        numRealisations = 200        
        
        datasetNames = []
        datasetNames.append(("abalone", numRealisations))
        datasetNames.append(("add10", numRealisations))
        datasetNames.append(("comp-activ", numRealisations))
        datasetNames.append(("concrete", numRealisations))
        datasetNames.append(("parkinsons-motor", numRealisations))
        datasetNames.append(("parkinsons-total", numRealisations))
        datasetNames.append(("pumadyn-32nh", numRealisations))
        datasetNames.append(("slice-loc", numRealisations))
        datasetNames.append(("winequality-red", numRealisations))
        datasetNames.append(("winequality-white", numRealisations))

        if withRealisations == True:
            return datasetNames
        else:
            datasetNames2 = []
            for datasetName, numReals in datasetNames:
                datasetNames2.append(datasetName)
            return datasetNames2

    @staticmethod
    def getRatschDatasets(withRealisations=False):
        datasetNames = []
        datasetNames.append(("banana", 100))
        datasetNames.append(("breast-cancer", 100))
        datasetNames.append(("diabetis", 100))
        datasetNames.append(("flare-solar", 100))
        datasetNames.append(("german", 100))
        datasetNames.append(("heart", 100))
        datasetNames.append(("image", 20))
        datasetNames.append(("ringnorm", 100))
        datasetNames.append(("splice", 20))
        datasetNames.append(("thyroid", 100))
        datasetNames.append(("titanic", 100))
        datasetNames.append(("twonorm", 100))
        datasetNames.append(("waveform", 100))

        if withRealisations == True:
            return datasetNames
        else:
            datasetNames2 = []
            for datasetName, numReals in datasetNames:
                datasetNames2.append(datasetName)
            return datasetNames2

    @staticmethod
    def getRatschDatasetsExt(withRealisations=False):
        datasetNames = []
        datasetNames.append(("banana", 20))
        datasetNames.append(("image", 20))
        datasetNames.append(("ringnorm", 20))
        datasetNames.append(("splice", 20))
        datasetNames.append(("twonorm", 20))
        datasetNames.append(("waveform", 20))

        if withRealisations == True:
            return datasetNames
        else:
            datasetNames2 = []
            for datasetName, numReals in datasetNames:
                datasetNames2.append(datasetName)
            return datasetNames2

    @staticmethod
    def bayesError(gridPoints, decisionGrid, pdfX, pdfY1X, pdfYminus1X):
        """
        Compute the bayes error by numerical intergration over a grid for a 2D 
        space. 

        :param gridPoints: A 1D array of points along the grid that we are interested in
        :param decisionsY: A 2D array of decision values on the grid.
        :param pdfX: The pdf of x for all points on the grid.
        :param pdfY1X: The pdf of x for all points on the grid.
        :param pdfYminus1X: The pdf of x for all points on the grid.
        """
        squareArea = (gridPoints[1]-gridPoints[0])**2

        #Now go through each square to compute an error
        error = 0
        for i in range(gridPoints.shape[0]-1):
            for j in range(gridPoints.shape[0]-1):
                decisionSum = decisionGrid[i, j]+decisionGrid[i+1, j]+decisionGrid[i, j+1]+decisionGrid[i+1, j+1]
                label = numpy.sign(decisionSum)

                px = (pdfX[i,j]+pdfX[i+1,j]+pdfX[i, j+1]+pdfX[i+1, j+1])/4

                if label == -1:
                    py1x = (pdfY1X[i,j]+pdfY1X[i+1,j]+pdfY1X[i, j+1]+pdfY1X[i+1, j+1])/4
                    error += px*py1x*squareArea
                else:
                    pyminus1x = (pdfYminus1X[i,j]+pdfYminus1X[i+1,j]+pdfYminus1X[i, j+1]+pdfYminus1X[i+1, j+1])/4
                    error += px*pyminus1x*squareArea

        return error