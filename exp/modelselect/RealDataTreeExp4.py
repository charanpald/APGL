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
sampleSizes = numpy.array([500])
foldsSet = numpy.arange(2, 13, 2)
alpha = 1.0

gammas = numpy.array(numpy.round(2**numpy.arange(1, 7.5, 0.5)-1), dtype=numpy.int)

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
            
            numpy.random.seed(21)
            trainInds = numpy.random.permutation(trainX.shape[0])[0:sampleSize]
            validX = trainX[trainInds,:]
            validY = trainY[trainInds]
                           
            betas = learner.learningRate(validX, validY, foldsSet, paramDict)       
            print(betas) 
            
            plt.plot(gammas, betas)
            plt.show()
        
    break 
