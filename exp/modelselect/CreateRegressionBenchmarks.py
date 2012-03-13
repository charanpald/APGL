#Process some UCI datasets to create a set of regression ones

import numpy
import csv
import os
from sklearn import preprocessing
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Sampling import Sampling 

numpy.set_printoptions(suppress=True, precision=3)


def preprocessSave(X, y, outputDir, idx):
    X = preprocessing.scale(X)
    y = preprocessing.scale(y)

    if not os.path.isdir(outputDir):
        os.mkdir(outputDir)

    trainIndsFileName = outputDir + "trainInds.txt"
    testIndsFileName = outputDir + "testInds.txt"
    trainWriter = csv.writer(open(trainIndsFileName, 'wb'), delimiter=',')
    testWriter = csv.writer(open(testIndsFileName, 'wb'), delimiter=',')

    print("Number of train and test inds: " + str(len(idx[0][0])) + " " + str(len(idx[0][1]))) 

    for trainInds, testInds in idx:
        trainWriter.writerow(trainInds)
        testWriter.writerow(testInds)
    
    dataFileName = outputDir + "data.txt"
    numpy.savetxt(dataFileName, numpy.c_[X, y], delimiter=",", fmt="%.6f")
    print("Examples shape is " + str(X.shape) + " and labels shape is " + str(y.shape))
    print("Saved data as " + dataFileName)
    print("-"*100 + "\n")
    
#Start with wines
def processSimpleDataset(name, numRealisations, split, ext=".csv", delimiter=",", usecols=None, skiprows=1, converters=None):
    numpy.random.seed(21)
    dataDir = PathDefaults.getDataDir() + "modelPenalisation/regression/"
    fileName = dataDir + name + ext
    
    print("Loading data from file " + fileName)
    outputDir = PathDefaults.getDataDir() + "modelPenalisation/regression/" + name + "/"

    XY = numpy.loadtxt(fileName, delimiter=delimiter, skiprows=skiprows, usecols=usecols, converters=converters)
    X = XY[:, :-1]
    y = XY[:, -1]
    idx = Sampling.shuffleSplit(numRealisations, X.shape[0], split)
    preprocessSave(X, y, outputDir, idx)

def processParkinsonsDataset(name):
    numpy.random.seed(21)
    dataDir = PathDefaults.getDataDir() + "modelPenalisation/regression/"
    fileName = dataDir + name + ".data"
    

    XY = numpy.loadtxt(fileName, delimiter=",", skiprows=1)
    inds = list(set(range(XY.shape[1])) - set([5, 6]))
    X = XY[:, inds]

    y1 = XY[:, 5]
    y2 = XY[:, 6]
    #We don't keep whole collections of patients
    split = 0.5

    numRealisations = 100
    idx = Sampling.shuffleSplit(numRealisations, X.shape[0], split)

    outputDir = PathDefaults.getDataDir() + "modelPenalisation/regression/" + name + "-motor/"
    preprocessSave(X, y1, outputDir, idx)
    
    outputDir = PathDefaults.getDataDir() + "modelPenalisation/regression/" + name + "-total/"
    preprocessSave(X, y2, outputDir, idx)

numRealisations = 100 
#processSimpleDataset("winequality-white", numRealisations, 2.0/3.0, delimiter=";")
#processSimpleDataset("winequality-red", numRealisations, 2.0/3.0, delimiter=";")
#processSimpleDataset("concrete", numRealisations, 0.3)
#processSimpleDataset("slice-loc", numRealisations, 0.5, usecols=tuple(range(1, 386)))
#processParkinsonsDataset("parkinsons")
#processSimpleDataset("abalone", numRealisations, 0.2, ext=".data", delimiter=",", skiprows=0) 
#processSimpleDataset("comp-activ", numRealisations, 0.3, ext=".data", delimiter=" ", usecols=tuple(range(1, 24)), skiprows=0) 
#processSimpleDataset("add10", numRealisations, 0.3, ext=".data", delimiter=" ", skiprows=0) 

processSimpleDataset("pumadyn-32nh", numRealisations, 0.4, ext=".data", delimiter=None, skiprows=0) 