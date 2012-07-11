"""
Try to figure out the learning rate 
"""
import logging 
import numpy 
import sys 
import multiprocessing 
from apgl.util.PathDefaults import PathDefaults 
from exp.modelselect.ModelSelectUtils import ModelSelectUtils 
from apgl.util.Sampling import Sampling
from exp.sandbox.predictors.DecisionTreeLearner import DecisionTreeLearner
import matplotlib.pyplot as plt 
from sklearn import linear_model 
from apgl.util.FileLock import FileLock

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.seterr(all="raise")
numpy.random.seed(45)
dataDir = PathDefaults.getDataDir() 
dataDir += "modelPenalisation/regression/"
outputDir = PathDefaults.getOutputDir() + "modelPenalisation/regression/CART/"

figInd = 0 

loadMethod = ModelSelectUtils.loadRegressDataset
datasets = ModelSelectUtils.getRegressionDatasets(True)

#sampleSizes = numpy.array([50, 100, 200])
sampleSizes = numpy.array([200])
foldsSet = numpy.arange(2, 13, 1)
alpha = 1.0

gammas = numpy.unique(numpy.array(numpy.round(2**numpy.arange(1, 7.25, 0.25)-1), dtype=numpy.int))

paramDict = {} 
paramDict["setGamma"] = gammas
numParams = paramDict["setGamma"].shape[0]

sampleMethod = Sampling.crossValidation
numProcesses = multiprocessing.cpu_count()
Cvs = numpy.array([1])

for datasetName, numRealisations in datasets:
    logging.debug("Dataset " + datasetName)
    
    
    learner = DecisionTreeLearner(criterion="mse", maxDepth=100, minSplit=1, pruneType="CART", processes=numProcesses)
    learner.setChunkSize(3)   
    
    outfileName = outputDir + datasetName + "Beta"

    for m in range(sampleSizes.shape[0]): 
        sampleSize = sampleSizes[m]
        logging.debug("Sample size " + str(sampleSize))
    
        penalties = numpy.zeros((foldsSet.shape[0], numParams))
        betas = numpy.zeros((gammas.shape[0], sampleSizes.shape[0]))
        
        for j in range(numRealisations):      
            logging.debug("Realisation: " + str(j))
            
            trainX, trainY, testX, testY = loadMethod(dataDir, datasetName, j)
            
            trainInds = numpy.arange(sampleSize)
            trainX = trainX[trainInds,:]
            trainY = trainY[trainInds]
                           
            for k in range(foldsSet.shape[0]): 
                folds = foldsSet[k]
                logging.debug("Folds " + str(folds))
                
                idx = sampleMethod(folds, trainX.shape[0])   
                
                #Now try penalisation
                resultsList = learner.parallelPen(trainX, trainY, idx, paramDict, Cvs)
                bestLearner, trainErrors, currentPenalties = resultsList[0]
                penalties[k, :] += currentPenalties
            
            for i in range(gammas.shape[0]): 
                inds = numpy.logical_and(numpy.isfinite(penalties[:, i]), penalties[:, i]>0)
                tempPenalties = penalties[:, i][inds]
                tempfoldsSet = numpy.array(foldsSet, numpy.float)[inds]                            
                
                if tempPenalties.shape[0] > 1: 
                    x = numpy.log((tempfoldsSet-1)/tempfoldsSet*sampleSize)
                    y = numpy.log(tempPenalties)+numpy.log(tempfoldsSet)   
                
                    clf = linear_model.LinearRegression()
                    clf.fit(numpy.array([x]).T, y)
                    betas[i, m] = clf.coef_[0]    
                    
            betas = -betas        
            print(betas) 
            
            plt.plot(gammas, betas[:, 0])
            plt.show()
            
            betas2 = learner.learningRate(trainX, trainY, foldsSet, paramDict)
            print(betas2)
    
    break 
