import sys
import os
import numpy
import logging
import multiprocessing 
from apgl.predictors.LibSVM import LibSVM
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Util import Util
from apgl.util.FileLock import FileLock
from apgl.util.Sampling import Sampling
from apgl.util.Evaluator import Evaluator
from apgl.modelselect.ModelSelectUtils import ModelSelectUtils, computeIdealPenalty, parallelPenaltyGridRbf
"""
Let's run cross validation and model penalisation over the a toy dataset.
"""

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.seterr(all="raise")
#util.log_to_stderr(util.SUBDEBUG)



def runToyExp(datasetNames, sampleSizes, foldsSet, cvScalings, sampleMethods, numProcesses, fileNameSuffix):
    dataDir = PathDefaults.getDataDir() + "modelPenalisation/toy/"
    outputDir = PathDefaults.getOutputDir() + "modelPenalisation/"

    svm = LibSVM()
    numCs = svm.getCs().shape[0]
    numGammas = svm.getGammas().shape[0]
    numMethods = 1+(1+cvScalings.shape[0])
    numParams = 2

    runIdeal = True
    runCv = True
    runVfpen = True

    for i in range(len(datasetNames)):
        datasetName = datasetNames[i][0]
        numRealisations = datasetNames[i][1]
        logging.debug("Learning using dataset " + datasetName)

        for s in range(len(sampleMethods)):
            sampleMethod = sampleMethods[s][1]
            outfileName = outputDir + datasetName + sampleMethods[s][0] + fileNameSuffix

            fileLock = FileLock(outfileName + ".npz")
            if not fileLock.isLocked() and not fileLock.fileExists():
                fileLock.lock()
                errors = numpy.zeros((numRealisations, len(sampleSizes), foldsSet.shape[0], numMethods))
                params = numpy.zeros((numRealisations, len(sampleSizes), foldsSet.shape[0], numMethods, numParams))
                errorGrids = numpy.zeros((numRealisations, len(sampleSizes), foldsSet.shape[0], numMethods, numCs, numGammas))
                approxGrids = numpy.zeros((numRealisations, len(sampleSizes), foldsSet.shape[0], numMethods, numCs, numGammas))
                idealGrids = numpy.zeros((numRealisations, len(sampleSizes), foldsSet.shape[0], numCs, numGammas))

                data = numpy.load(dataDir + datasetName + ".npz")
                gridPoints, trainX, trainY, pdfX, pdfY1X, pdfYminus1X = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"], data["arr_4"], data["arr_5"]

                #We form a test set from the grid points
                testX = numpy.zeros((gridPoints.shape[0]**2, 2))
                for m in range(gridPoints.shape[0]):
                    testX[m*gridPoints.shape[0]:(m+1)*gridPoints.shape[0], 0] = gridPoints
                    testX[m*gridPoints.shape[0]:(m+1)*gridPoints.shape[0], 1] = gridPoints[m]

                for j in range(numRealisations):
                    Util.printIteration(j, 1, numRealisations, "Realisation: ")

                    for k in range(sampleSizes.shape[0]):
                        sampleSize = sampleSizes[k]
                        for m in range(foldsSet.shape[0]):
                            folds = foldsSet[m]
                            logging.debug("Using sample size " + str(sampleSize) + " and " + str(folds) + " folds")
                            perm = numpy.random.permutation(trainX.shape[0])
                            trainInds = perm[0:sampleSize]
                            validX = trainX[trainInds, :]
                            validY = trainY[trainInds]

                            svm = LibSVM(processes=numProcesses)
                            #Find ideal penalties
                            if runIdeal:
                                logging.debug("Finding ideal grid of penalties")
                                idealGrids[j, k, m, :, :] = parallelPenaltyGridRbf(svm, validX, validY, testX, gridPoints, pdfX, pdfY1X, pdfYminus1X)

                            #Cross validation
                            if runCv:
                                logging.debug("Running V-fold cross validation")
                                methodInd = 0
                                idx = sampleMethod(folds, validY.shape[0])
                                if sampleMethod == Sampling.bootstrap:
                                    bootstrap = True
                                else:
                                    bootstrap = False

                                bestSVM, cvGrid = svm.parallelVfcvRbf(validX, validY, idx, True, bootstrap)
                                predY, decisionsY = bestSVM.predict(testX, True)
                                decisionGrid = numpy.reshape(decisionsY, (gridPoints.shape[0], gridPoints.shape[0]), order="F")
                                errors[j, k, m, methodInd] = ModelSelectUtils.bayesError(gridPoints, decisionGrid, pdfX, pdfY1X, pdfYminus1X)
                                params[j, k, m, methodInd, :] = numpy.array([bestSVM.getC(), bestSVM.getKernelParams()])
                                errorGrids[j, k, m, methodInd, :, :] = cvGrid

                            #v fold penalisation
                            if runVfpen:
                                logging.debug("Running penalisation")
                                #BIC penalisation
                                Cv = float((folds-1) * numpy.log(validX.shape[0])/2)
                                tempCvScalings = cvScalings*(folds-1)
                                tempCvScalings = numpy.insert(tempCvScalings, 0, Cv)

                                #Use cross validation
                                idx = sampleMethod(folds, validY.shape[0])
                                svmGridResults = svm.parallelVfPenRbf(validX, validY, idx, tempCvScalings)

                                for n in range(len(tempCvScalings)):
                                    bestSVM, trainErrors, approxGrid = svmGridResults[n]
                                    methodInd = n+1
                                    predY, decisionsY = bestSVM.predict(testX, True)
                                    decisionGrid = numpy.reshape(decisionsY, (gridPoints.shape[0], gridPoints.shape[0]), order="F")
                                    errors[j, k, m, methodInd] = ModelSelectUtils.bayesError(gridPoints, decisionGrid, pdfX, pdfY1X, pdfYminus1X)
                                    params[j, k, m, methodInd, :] = numpy.array([bestSVM.getC(), bestSVM.getKernelParams()])
                                    errorGrids[j, k, m, methodInd, :, :] = trainErrors + approxGrid
                                    approxGrids[j, k, m, methodInd, :, :] = approxGrid


                meanErrors = numpy.mean(errors, 0)
                print(meanErrors)

                meanParams = numpy.mean(params, 0)
                print(meanParams)

                meanErrorGrids = numpy.mean(errorGrids, 0)
                stdErrorGrids = numpy.std(errorGrids, 0)

                meanIdealGrids = numpy.mean(idealGrids, 0)
                stdIdealGrids = numpy.std(idealGrids, 0)

                meanApproxGrids = numpy.mean(approxGrids, 0)
                stdApproxGrids = numpy.std(approxGrids, 0)

                numpy.savez(outfileName, errors, params, meanErrorGrids, stdErrorGrids, meanIdealGrids, stdIdealGrids, meanApproxGrids, stdApproxGrids)
                logging.debug("Saved results as file " + outfileName + ".npz")
                fileLock.unlock()
            else:
                logging.debug("Results already computed")

    logging.debug("All done!")


sampleSizes = numpy.array([50, 100, 200])
#sampleSizes = numpy.array([50])
foldsSet = numpy.arange(2, 13, 2)
#foldsSet = numpy.array([5])
cvScalings = numpy.arange(0.8, 1.81, 0.2)
sampleMethods = [("CV", Sampling.crossValidation), ("BS", Sampling.bootstrap), ("SS", Sampling.shuffleSplit)]
numProcesses = multiprocessing.cpu_count()

logging.debug("Running " + str(numProcesses) + " processes")
logging.debug("Process id: " + str(os.getpid()))

datasetNames = []
datasetNames.append(("toyData", 20))

fileNameSuffix = "Results"
runToyExp(datasetNames, sampleSizes, foldsSet, cvScalings, sampleMethods, numProcesses, fileNameSuffix)

#Now run some extended results
sampleSizes = numpy.array([50, 100, 200, 400, 800])
#sampleSizes = numpy.array([50])
foldsSet = numpy.arange(10, 51, 10)
#foldsSet = numpy.array([5])

datasetNames = []
datasetNames.append(("toyData", 10))

fileNameSuffix = "ResultsExt"
runToyExp(datasetNames, sampleSizes, foldsSet, cvScalings, sampleMethods, numProcesses, fileNameSuffix)
