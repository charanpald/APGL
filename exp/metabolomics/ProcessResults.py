"""
Plot the ROC curves for the metabolomics experiment. 
"""
import sys
import numpy 
import logging
import matplotlib.pyplot as plt
from apgl.util.Util import Util
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Latex import Latex 

logging.basicConfig(stream=sys.stdout, level=logging.WARN)
resultsDir = PathDefaults.getOutputDir() + "metabolomics/"
figureDir = resultsDir + "figures/"

labelNames = ["Testosterone.val_0", "Testosterone.val_1", "Testosterone.val_2"]
labelNames.extend(["Cortisol.val_0", "Cortisol.val_1", "Cortisol.val_2"])
labelNames.extend(["IGF1.val_0", "IGF1.val_1", "IGF1.val_2"])

labelNames2 = ["Testosterone.val", "Cortisol.val", "IGF1.val"]

algorithmNames = ["TreeRank"]
#algorithmNames = ["TreeRankForest"]
leafRankNames = ["CART", "SVM", "RBF-SVM", "LinearSVM-PCA"]
#leafRankNames = ["CARTF", "SVMF", "RBF-SVMF"]
dataTypes = ["raw_std", "log", "opls"]
#dataTypes = []

Ns = [10, 25, 50, 75, 100]
dataTypes.append("Db4")
dataTypes.append("Db8")
dataTypes.append("Haar")

plotInd = 0
plotStyles = ['ko-', 'kx-', 'k+-', 'k.-', 'k*-']

numMethods = len(leafRankNames)*len(dataTypes)*len(algorithmNames)
meanAUCs = numpy.zeros((numMethods, len(labelNames)))
stdAUCs = numpy.zeros((numMethods, len(labelNames)))

results = 0 
missingResults = 0
rowNames = []
fileNamesArray = numpy.zeros((numMethods, len(labelNames))).tolist()

def loadResultsFile(fileName):
    resultsDict = Util.loadPickle(fileName)
    allMetrics = resultsDict["allMetrics"]
    bestAUCs = allMetrics[2]
    trainROCs = allMetrics[1]
    testROCs = allMetrics[3]
    logging.debug("Mean AUC is " + str(numpy.mean(bestAUCs)))
    logging.debug("Best parameters are " + str(resultsDict["bestParams"]))

    meanAUC = numpy.mean(bestAUCs)
    stdAUC = numpy.std(bestAUCs)

    return meanAUC, stdAUC, trainROCs, testROCs

for m in range(len(algorithmNames)):
    algorithmName = algorithmNames[m]

    for i in range(len(labelNames)):
        labelName = labelNames[i]

        for j in range(len(leafRankNames)):
            leafRank = leafRankNames[j]

            for k in range(len(dataTypes)):
                dataType = dataTypes[k]
                rowNames.append((leafRank + " " + dataType).ljust(10))
                rowInd = m*len(leafRankNames)*len(dataTypes) + j*len(dataTypes) + k

                if dataType == "Db4" or dataType == "Db8" or dataType == "Haar":
                    dataSubTypes = [dataType + "-" + str(N) for N in Ns]
                    dataSubTypes.append(dataType)

                    for dataSubType in dataSubTypes:
                        fileName = algorithmName + "-" + labelName + "-" + leafRank + "-" + dataSubType + ".dat"
                        fullFileName = resultsDir + fileName
                
                        try:
                            meanAUC, stdAUC, trainROCs, testROCs = loadResultsFile(fullFileName)

                            if meanAUC >= meanAUCs[rowInd, i]:
                                meanAUCs[rowInd, i], stdAUCs[rowInd, i], fileNamesArray[rowInd][i]  = meanAUC, stdAUC, fileName
                            results += 1
                        except IOError as e:
                            missingResults += 1
                else:
                    try:
                        fileName = algorithmName + "-" + labelName + "-" + leafRank + "-" + dataType + ".dat"
                        fullFileName = resultsDir + fileName
                        meanAUC, stdAUC, trainROCs, testROCs = loadResultsFile(fullFileName)
                        meanAUCs[rowInd, i], stdAUCs[rowInd, i], fileNamesArray[rowInd][i]  = meanAUC, stdAUC, fileName
                        results += 1
                    except IOError as e:
                        missingResults += 1

print("Found " + str(results) + ", missing " + str(missingResults) + " results\n")

#Plot the best ROC curves
for i in range(len(labelNames)):
    labelName = labelNames[i]
    ind = numpy.argmax(meanAUCs[:, i])
    
    fileName = fileNamesArray[ind][i]
    print(fileName + " AUC= " + str(meanAUCs[ind, i]))
    fullFileName = resultsDir + fileName
    resultsDict = Util.loadPickle(fullFileName)
    allMetrics = resultsDict["allMetrics"]
    trainROCs = allMetrics[1]
    testROCs = allMetrics[3]

    for m in range(len(trainROCs)):
        plt.figure(plotInd)
        plt.title(fileName)
        plt.plot(trainROCs[m][0], trainROCs[m][1], "k")
        plt.plot(testROCs[m][0], testROCs[m][1], "r")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.savefig(figureDir + labelName.replace(".", "_") + "-ROC.eps")
        
    plotInd += 1 


#The last column is the mean value
meanMeanAUCs = numpy.mean(meanAUCs, 1)
meanStdAUCs = numpy.mean(stdAUCs, 1)
meanAUCs = numpy.c_[meanAUCs, meanMeanAUCs]
stdAUCs = numpy.c_[stdAUCs, meanStdAUCs]

print("\n")
print(Latex.listToRow(labelNames))
print(Latex.addRowNames(rowNames, Latex.array2DsToRows(meanAUCs, stdAUCs, 2)))


#----------------- Results for RankBoost and RankSVM  ----------------------------------

algorithmNames = ["RankBoost", "RankSVM"]
numMethods = len(dataTypes)*len(algorithmNames)
rowNames = numpy.zeros(numMethods, "a20")
meanAUCs = numpy.zeros((numMethods, len(labelNames)))
stdAUCs = numpy.zeros((numMethods, len(labelNames)))

for m in range(len(algorithmNames)):
    algorithmName = algorithmNames[m]

    for i in range(len(labelNames)):
        labelName = labelNames[i]

        for k in range(len(dataTypes)):
            dataType = dataTypes[k]
            rowInd = m*len(dataTypes) + k
            rowNames[rowInd] = algorithmName + " " + (dataType).ljust(10)

            if dataType == "Db4" or dataType == "Db8" or dataType == "Haar":
                dataSubTypes = [dataType + "-" + str(N) for N in Ns]
                dataSubTypes.append(dataType)

                for dataSubType in dataSubTypes:
                    fileName = algorithmName + "-" + labelName + "-" + dataSubType + ".dat"
                    fullFileName = resultsDir + fileName

                    try:
                        meanAUC, stdAUC, trainROCs, testROCs = loadResultsFile(fullFileName)

                        if meanAUC >= meanAUCs[rowInd, i]:
                            meanAUCs[rowInd, i], stdAUCs[rowInd, i], fileNamesArray[rowInd][i]  = meanAUC, stdAUC, fileName
                        results += 1
                    except IOError as e:
                        missingResults += 1
            else:
                try:
                    fileName = algorithmName + "-" + labelName + "-" + dataType + ".dat"
                    fullFileName = resultsDir + fileName
                    meanAUC, stdAUC, trainROCs, testROCs = loadResultsFile(fullFileName)
                    meanAUCs[rowInd, i], stdAUCs[rowInd, i], fileNamesArray[rowInd][i]  = meanAUC, stdAUC, fileName
                    results += 1
                except IOError as e:
                    missingResults += 1

print("Found " + str(results) + ", missing " + str(missingResults) + " results\n")

meanMeanAUCs = numpy.mean(meanAUCs, 1)
meanStdAUCs = numpy.mean(stdAUCs, 1)
meanAUCs = numpy.c_[meanAUCs, meanMeanAUCs]
stdAUCs = numpy.c_[stdAUCs, meanStdAUCs]

print("\n")
print(Latex.listToRow(labelNames))
print(Latex.addRowNames(rowNames, Latex.array2DsToRows(meanAUCs, stdAUCs, 2)))

print(numpy.amax(meanAUCs, 0))

#---------------- Ordinal Regression Results -----------------------------

regressionMethods = ["lasso", "svr_rbf", "svr_poly"]
indVars = 3
numMethods = len(regressionMethods)*len(dataTypes)
meanAUCs = numpy.zeros((numMethods, len(labelNames)))
stdAUCs = numpy.zeros((numMethods, len(labelNames)))

results = 0 
missingResults = 0 
meanSqErrs = numpy.zeros((numMethods, len(labelNames2)))
stdSqErrs = numpy.zeros((numMethods, len(labelNames2)))

for i in range(len(labelNames2)):
    labelName = labelNames2[i]
    for j in range(len(regressionMethods)):
        regressionMethod = regressionMethods[j]
        for k in range(len(dataTypes)):
            dataType = dataTypes[k]
            rowInd = j*len(dataTypes) + k

            if dataType == "Db4" or dataType == "Db8" or dataType == "Haar":
                dataSubTypes = [dataType + "-" + str(N) for N in Ns]
                dataSubTypes.append(dataType)

                for dataSubType in dataSubTypes:
                    fileName = labelName + "-" + regressionMethod + "-" + dataSubType + ".dat"
                    fullFileName = resultsDir + fileName

                    try:
                        (allMetrics, rankMetrics, paramStrList) = Util.loadPickle(fullFileName)
                        meanSqErrs[rowInd, i] = numpy.mean(allMetrics[0])
                        stdSqErrs[rowInd, i] = numpy.std(allMetrics[0])

                        for m in range(indVars):
                            colInd = i*indVars+m
                            meanAUCs[rowInd, colInd] = numpy.mean(rankMetrics[:, m])
                            stdAUCs[rowInd, colInd] = numpy.std(rankMetrics[:, m])

                        results += 1
                    except IOError as e:
                        missingResults += 1
            else:
                fileName = labelName + "-" + regressionMethod + "-" + dataType + ".dat"
                fullFileName = resultsDir + fileName

                try:
                    (allMetrics, rankMetrics, paramStrList) = Util.loadPickle(fullFileName)
                    meanSqErrs[rowInd, i] = numpy.mean(allMetrics[0])
                    stdSqErrs[rowInd, i] = numpy.std(allMetrics[0])

                    for m in range(indVars):
                        colInd = i*indVars+m
                        meanAUCs[rowInd, colInd] = numpy.mean(rankMetrics[:, m])
                        stdAUCs[rowInd, colInd] = numpy.std(rankMetrics[:, m])

                    results += 1
                except IOError as e:
                    missingResults += 1

print("Found " + str(results) + ", missing " + str(missingResults) + " results\n\n")

rowNames = []
for regressionMethod in regressionMethods:
    for dataType in dataTypes:
        rowNames.append((regressionMethod + " " + dataType).ljust(10))

meanMeanAUCs = numpy.mean(meanAUCs, 1)
meanStdAUCs = numpy.mean(stdAUCs, 1)
meanAUCs = numpy.c_[meanAUCs, meanMeanAUCs]
stdAUCs = numpy.c_[stdAUCs, meanStdAUCs]

print(Latex.listToRow(labelNames))
print(Latex.addRowNames(rowNames, Latex.array2DsToRows(meanAUCs, stdAUCs, 2)))

print(numpy.max(meanAUCs, 0))

#---------------- Mean Squared Error Regression Results -----------------------------

#print(Latex.listToRow(labelNames2))
#print(Latex.addRowNames(rowNames, Latex.array2DsToRows(meanSqErrs, stdSqErrs, 4)))

#Plot results
#plt.show()