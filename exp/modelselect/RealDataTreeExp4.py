"""
Try to figure out the learning rate 
"""
import logging 
import numpy 
import sys 
import multiprocessing 
import scipy.stats 
from apgl.util.PathDefaults import PathDefaults 
from exp.modelselect.ModelSelectUtils import ModelSelectUtils 
from apgl.util.Sampling import Sampling
from exp.sandbox.predictors.DecisionTreeLearner import DecisionTreeLearner
import matplotlib.pyplot as plt 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.seterr(all="raise")
numpy.random.seed(45)
dataDir = PathDefaults.getDataDir() 
dataDir += "modelPenalisation/regression/"

figInd = 0 

loadMethod = ModelSelectUtils.loadRegressDataset
datasets = ModelSelectUtils.getRegressionDatasets(True)

sampleSizes = numpy.array([50, 100, 200])
foldSizes = numpy.arange(2, 13, 2)
alpha = 1.0

gammas = numpy.array(numpy.round(2**numpy.arange(1, 7.5, 0.5)-1), dtype=numpy.int)

paramDict = {} 
paramDict["setGamma"] = numpy.array([3], dtype=numpy.int)
numParams = paramDict["setGamma"].shape[0]

sampleMethod = Sampling.crossValidation
numProcesses = multiprocessing.cpu_count()
Cvs = numpy.array([1])

outputDir = PathDefaults.getOutputDir() + "modelPenalisation/regression/CART/"

#datasets = [datasets[0]]

for datasetName, numRealisations in datasets:
    logging.debug("Dataset " + datasetName)
    betas = numpy.zeros((gammas.shape[0], sampleSizes.shape[0]))
    
    learner = DecisionTreeLearner(criterion="mse", maxDepth=100, minSplit=1, pruneType="CART", processes=numProcesses)
    learner.setChunkSize(3)    
    
    for m in range(sampleSizes.shape[0]): 
        sampleSize = sampleSizes[m]
        logging.debug("Sample size " + str(sampleSize))
    
        for i in range(gammas.shape[0]): 
            gamma = gammas[i]
            logging.debug("Gamma " + str(gamma))
            paramDict["setGamma"] = numpy.array([gamma], dtype=numpy.int)
            meanErrors = numpy.zeros(foldSizes.shape[0])   
            meanPenalties = numpy.zeros(foldSizes.shape[0])
        
            numRealisations = 20
            
            k = 0 
            for folds in foldSizes: 
                logging.debug("Folds " + str(folds))
                errors = numpy.zeros(numRealisations)
                
                meanAllErrors = numpy.zeros(numParams) 
                meanTrainError = numpy.zeros(numParams)
                        
                for j in range(numRealisations):
                    print("")
                    logging.debug("Realisation: " + str(j))
                    trainX, trainY, testX, testY = loadMethod(dataDir, datasetName, j)
                    
                    trainInds = numpy.random.permutation(trainX.shape[0])[0:sampleSize]
                    trainX = trainX[trainInds,:]
                    trainY = trainY[trainInds]
                    
                    idx = sampleMethod(folds, trainX.shape[0])        
                    
                    #Now try penalisation
                    resultsList = learner.parallelPen(trainX, trainY, idx, paramDict, Cvs)
                    bestLearner, trainErrors, currentPenalties = resultsList[0]
                    meanPenalties[k] += currentPenalties
                    meanTrainError += trainErrors
                    predY = bestLearner.predict(testX)
                    meanErrors[k] += bestLearner.getMetricMethod()(testY, predY)
        
                k+= 1
                
            numRealisations = float(numRealisations)
            meanErrors /=  numRealisations 
            meanPenalties /=  numRealisations 
        
            inds = numpy.logical_and(numpy.isfinite(meanPenalties), meanPenalties>0)
            meanErrors = meanErrors[inds]
            meanPenalties = meanPenalties[inds]
            tempFoldSizes = numpy.array(foldSizes, numpy.float)[inds]    
            
            print(meanPenalties)        
            
            if meanPenalties.shape[0] != 0: 
                x = numpy.log((tempFoldSizes-1)/tempFoldSizes*sampleSize)
                y = numpy.log(meanPenalties)+numpy.log(tempFoldSizes)    
            
                print(x, y)    
            
                betas[i, m], intercept, r, p, err = scipy.stats.linregress(x, y)
                print(gamma, betas)        
            
            """
            plt.figure(1)
            plt.plot(x, y, label="Penalty")
            plt.xlabel("log(sampleSize*folds)")
            plt.ylabel("log(meanPenalties)")   
            plt.legend()
                
            plt.show()
            """
        
    print(-betas) 
    
    outfileName = outputDir + datasetName + "Beta"
    numpy.savez(outfileName, -betas)
    logging.debug("Saved results as file " + outfileName + ".npz")