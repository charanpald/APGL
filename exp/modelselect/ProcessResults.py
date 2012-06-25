import numpy
import logging
import sys 
import scipy.stats 
from apgl.util.Latex import Latex
from apgl.util.PathDefaults import PathDefaults
from exp.modelselect.ModelSelectUtils import ModelSelectUtils
import matplotlib.pyplot as plt

#Produce latex tables from the benchmark results
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def getWins(errors):
    """
    Want to compute the number of wins/ties/losses versus CV 
    """
    #Shape is realisations, len(sampleSizes), foldsSet.shape[0], numMethods
    winsShape = list(errors.shape[1:]) 
    winsShape.append(3)
    stdWins = numpy.zeros(winsShape, numpy.int)
    p = 0.1
    
    meanErrors = numpy.mean(errors, 0)
   
    for i in range(len(sampleSizes)):
        #Now figure out if wins are significant 
        for j in range(foldsSet.shape[0]): 
            for k in range(meanErrors.shape[2]): 
                s1 = errors[:, i, j, 0]
                s2 = errors[:, i, j, k]
                
                s1Mean = numpy.mean(s1)
                s2Mean = numpy.mean(s2)                
                
                t, prob = scipy.stats.ttest_ind(s1, s2)
                if prob < p: 
                    if s1Mean > s2Mean: 
                        stdWins[i, j, k, 2] = 1 
                    elif s1Mean < s2Mean:
                        stdWins[i, j, k, 0] = 1
                else: 
                    stdWins[i, j, k, 1] = 1 
                    
    return stdWins

def getRowNames(cvScalings):
    rowNames = [""]
    for j in range(sampleSizes.shape[0]):
        rowNames.append("Std" + " $m=" + str(sampleSizes[j]) + "$")
        rowNames.append("PenBIC" + " $m=" + str(sampleSizes[j]) + "$")
        for k in range(cvScalings.shape[0]):
            rowNames.append("PenVF" + " $m=" + str(sampleSizes[j]) + "$ $\\alpha=" + str(cvScalings[k]) + "$")
    return rowNames 

def getLatexTable(measures, cvScalings):
    rowNames = getRowNames(cvScalings)
    table = Latex.array1DToRow(foldsSet) + "\\\\ \n"

    for j in range(sampleSizes.shape[0]):
        meanMeasures = numpy.mean(measures, 0)
        stdMeasures = numpy.std(measures, 0)
        table += Latex.array2DToRows(meanMeasures[j, :, :].T, stdMeasures[j, :, :].T) + "\n"

    table = Latex.addRowNames(rowNames, table)
    return table, meanMeasures, stdMeasures

def summary(datasetNames, sampleSizes, foldsSet, cvScalings, sampleMethods, fileNameSuffix):
    """
    Print the errors for all results plus a summary. 
    """
    numMethods = (1+(cvScalings.shape[0]+1))
    numDatasets = len(datasetNames)
    overallErrors = numpy.zeros((numDatasets, len(sampleMethods), sampleSizes.shape[0], foldsSet.shape[0], numMethods))
    overallStdWins = numpy.zeros((len(sampleMethods), len(sampleSizes), foldsSet.shape[0], numMethods, 3), numpy.int)
    overallErrorsPerSampMethod = numpy.zeros((numDatasets, len(sampleMethods), len(sampleSizes), numMethods), numpy.float)
    
    table1 = ""
    table2 = ""
    table3 = ""

    for i in range(len(datasetNames)):
        table3Error = numpy.zeros((2, len(sampleMethods)))   
        table3Stds = numpy.zeros((2, len(sampleMethods)))   
        
        for j in range(len(sampleMethods)):
            print("="*50 + "\n" + datasetNames[i] + "-" + sampleMethods[j] + "\n" + "="*50 )
            outfileName = outputDir + datasetNames[i] + sampleMethods[j] + fileNameSuffix + ".npz"
            data = numpy.load(outfileName)

            errors = data["arr_0"]
            params = data["arr_1"]
            meanErrorGrids = data["arr_2"]
            stdErrorGrids = data["arr_3"]
            meanIdealGrids = data["arr_4"]
            stdIdealGrids = data["arr_5"]
            meanApproxGrids = data["arr_6"]
            stdApproxGrids = data["arr_7"]

            #print(params.mean(0))
            #print(meanErrorGrids[0, 0, 0, :])            
            
            errorTable, meanErrors, stdErrors = getLatexTable(errors, cvScalings)

            wins = getWins(errors)
            excessError = numpy.zeros(errors.shape)

            for k in range(errors.shape[1]):
                excessError[:, k, :, :] = errors[:, k, :, :] - numpy.tile(errors[:, k, :, 0, numpy.newaxis], (1, 1, numMethods))

            meanExcessError = numpy.mean(excessError, 0)
            stdExcessError = numpy.std(excessError, 0)
            excessErrorTable, meanExcessErrors, stdExcessErrors = getLatexTable(excessError, cvScalings)
            #print(excessErrorTable)
            #print(-meanExcessError > stdExcessError)
            #lossInd = 2
            #print(numpy.c_[wins[0, :, :, lossInd], wins[1, :, :, lossInd], wins[2, :, :, lossInd]].T)

            overallErrorsPerSampMethod[i, j, :, :] = numpy.mean(meanErrors, 1)
            overallErrors[i, j, :, :, :] = meanExcessError
            overallStdWins[j, :] += wins
            print(errorTable)
            #print("Min error is: " + str(numpy.min(meanErrors)))
            #print("Max error is: " + str(numpy.max(meanErrors)))
            #print("Mean error is: " + str(numpy.mean(meanErrors)) + "\n")
            
            #This is a table with V=10, alpha=1 and CV sampling 
            """
            print(meanErrors[0, 4, 0])
            table1Error = numpy.zeros(len(sampleSizes)*2)
            table1Std = numpy.zeros(len(sampleSizes)*2)
            for  k in range(len(sampleSizes)):
                table1Error[k*2] = meanErrors[k, 4, 0]
                table1Error[k*2+1] = meanErrors[k, 4, 3]
                table1Std[k*2] = stdErrors[k, 4, 0]
                table1Std[k*2+1] = stdErrors[k, 4, 3]
                
            if j == 0: 
                table1 += datasetNames[i] + " & " + Latex.array2DToRows(numpy.array([table1Error]), numpy.array([table1Std])) + "\n"
            
            tenFoldIndex = 4            
            
            #See how alpha varies with V=10, CV sampling 
            table2Error = numpy.zeros(range(numMethods-2))
            table2Std = numpy.zeros(range(numMethods-2))
            for s in range(len(sampleSizes)): 
                table2Error = meanErrors[s, tenFoldIndex, 2:]
                table2Std = stdErrors[s, tenFoldIndex, 2:]
            
                if j == 0: 
                    table2 += datasetNames[i] + " $m=" + str(sampleSizes[s]) + "$ & " + Latex.array2DToRows(numpy.array([table2Error]), numpy.array([table2Std])) + "\n"

            #See how each sample method effects CV and pen alpha=1
            fourFoldIndex = 4  
            hundredMIndex = 1            
            
            table3Error[0, j] = meanErrors[hundredMIndex, fourFoldIndex, 0]
            table3Error[1, j] = meanErrors[hundredMIndex, fourFoldIndex, 3]
            table3Stds[0, j] = stdErrors[hundredMIndex, fourFoldIndex, 0]
            table3Stds[1, j] = stdErrors[hundredMIndex, fourFoldIndex, 3]
            """

        table3 +=  Latex.addRowNames([datasetNames[i] + " Std ", datasetNames[i] + " PenVF "], Latex.array2DToRows(table3Error, table3Stds))            
            
        datasetMeanErrors = Latex.listToRow(sampleMethods) + "\n"

        for j in range(len(sampleSizes)):
            datasetMeanErrors += Latex.array2DToRows(overallErrorsPerSampMethod[i, :, j, :].T) + "\n"

        datasetMeanErrors = Latex.addRowNames(getRowNames(cvScalings), datasetMeanErrors)
        print(datasetMeanErrors)
     
    print("="*50 + "\n" + "Sliced Tables" + "\n" + "="*50)   
    
    print(table1 + "\n")
    print(table2 + "\n")
    print(table3)
     
    print("="*50 + "\n" + "Overall" + "\n" + "="*50)

    overallMeanErrors = numpy.mean(overallErrors, 0)
    overallStdErrors = numpy.std(overallErrors, 0)

    for i in range(len(sampleMethods)):
        print("-"*20 + sampleMethods[i] + "-"*20)
        overallErrorTable = Latex.array1DToRow(foldsSet) + "\\\\ \n"
        overallWinsTable = Latex.array1DToRow(foldsSet) + " & Total & "  +Latex.array1DToRow(foldsSet) + " & Total \\\\ \n"

        rowNames = getRowNames(cvScalings)

        for j in range(sampleSizes.shape[0]):
            overallErrorTable += Latex.array2DToRows(overallMeanErrors[i, j, :, :].T, overallStdErrors[i, j, :, :].T, bold=overallMeanErrors[i, j, :, :].T<0) + "\n"

            tiesWins = numpy.r_[overallStdWins[i, j, :, :, 0], overallStdWins[i, j, :, :, 1], overallStdWins[i, j, :, :, 2]]            
            
            overallWinsTable += Latex.array2DToRows(tiesWins.T) + "\n"

        overallErrorTable = Latex.addRowNames(rowNames, overallErrorTable)
        overallWinsTable = Latex.addRowNames(rowNames, overallWinsTable)

        print(Latex.latexTable(overallWinsTable, "Wins for " + sampleMethods[i], True))
        print(Latex.latexTable(overallErrorTable.replace("0.", "."), "Excess errors for " + sampleMethods[i], True))
        #print(overallWinsTable)
        #print(overallErrorTable)

    #Now print the mean errors for all datasets
    datasetMeanErrors = Latex.listToRow(sampleMethods) + "\n"
    overallErrorsPerSampMethod = numpy.mean(overallErrorsPerSampMethod[:, :, :, :], 0)

    for j in range(len(sampleSizes)):
        datasetMeanErrors += Latex.array2DToRows(overallErrorsPerSampMethod[:, j, :].T) + "\n"

    datasetMeanErrors = Latex.addRowNames(getRowNames(cvScalings), datasetMeanErrors)
    print(datasetMeanErrors)

def plotResults(datasetName, sampleSizes, foldsSet, cvScalings, sampleMethods, fileNameSuffix):
    """
    Plots the errors for a particular dataset on a bar graph. 
    """

    for k in range(len(sampleMethods)):
        outfileName = outputDir + datasetName + sampleMethods[k] + fileNameSuffix + ".npz"
        data = numpy.load(outfileName)

        errors = data["arr_0"]
        meanMeasures = numpy.mean(errors, 0)

        for i in range(sampleSizes.shape[0]):
            plt.figure(k*len(sampleMethods) + i)
            plt.title("n="+str(sampleSizes[i]) + " " + sampleMethods[k])

            for j in range(errors.shape[3]):
                plt.plot(foldsSet, meanMeasures[i, :, j])
                plt.xlabel("Folds")
                plt.ylabel('Error')

            labels = ["VFCV", "BIC"]
            labels.extend(["VFP s=" + str(x) for x in cvScalings])
            plt.legend(tuple(labels))
    plt.show()

#outputDir = PathDefaults.getOutputDir() + "modelPenalisation/regression/SVR/"
#outputDir = PathDefaults.getOutputDir() + "modelPenalisation/regression/DTRP/"
outputDir = PathDefaults.getOutputDir() + "modelPenalisation/regression/CART/old/"

#First output the fine grained results 
sampleSizes = numpy.array([50, 100, 200])
#sampleMethods = ["CV","SS", "SS66", "SS90"]
#sampleMethods = ["SS66", "SS90"]
sampleMethods = ["CV", "RCV"]
cvScalings = numpy.arange(0.6, 1.61, 0.2)
foldsSet = numpy.arange(2, 13, 2)
#datasetNames = ModelSelectUtils.getRatschDatasets()
datasetNames = ModelSelectUtils.getRegressionDatasets()
fileNameSuffix = 'Results'
summary(datasetNames, sampleSizes, foldsSet, cvScalings, sampleMethods, fileNameSuffix)

#plotResults("waveform", sampleSizes, foldsSet, cvScalings, sampleMethods, fileNameSuffix)

#Now run some extended results
sampleSizes = numpy.array([25, 50, 100])
foldsSet = numpy.arange(2, 13, 2)
cvScalings = numpy.arange(0.6, 1.61, 0.2)
sampleMethods = ["CV"]
datasetNames = ModelSelectUtils.getRegressionDatasets()

fileNameSuffix = "ResultsExt"
summary(datasetNames, sampleSizes, foldsSet, cvScalings, sampleMethods, fileNameSuffix)

