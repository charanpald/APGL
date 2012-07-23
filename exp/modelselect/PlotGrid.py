"""
Plot the error grid of the datasets.
"""
import numpy
import logging
import sys
import matplotlib.pyplot as plt
from apgl.util.PathDefaults import PathDefaults
from exp.modelselect.ModelSelectUtils import ModelSelectUtils
from operator import itemgetter

numpy.set_printoptions(linewidth=200)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def plotErrorGrid(plt, Cs, gammas, errorGrid, title, regression): 
    if regression: 
        errorGrid = numpy.min(errorGrid, 1)
    
    plt.contourf(numpy.log2(Cs), numpy.log2(gammas), errorGrid, 100, antialiased=True)
    plt.colorbar()
    plt.xlabel("log(C)")
    plt.ylabel("log(gamma)")
    plt.title(title)
    
    #error10 = numpy.array([ (numpy.log2(Cs[ii]),numpy.log2(gammas[jj]),errorGrid[jj,ii].min()) for ii in range(Cs.shape[0]) for jj in range(gammas.shape[0]) ])
    #top10 = numpy.array(sorted(error10,key=itemgetter(2)))[0:10, :]
    #plt.scatter(top10[:, 0], top10[:, 1])

def plotErrorGridCART(plt, gammas, errorGrid, title):    
    plt.plot(numpy.log2(gammas), errorGrid, label=title)
    plt.xlabel("log(gamma)")
    plt.ylabel("error")
    plt.title(title)
    
def plotGridsCART(datasetNames, sampleSizes, foldsSet, cvScalings, sampleMethods, fileNameSuffix, foldsInd=1): 
    figInd = 0 
    for i in range(len(datasetNames)):
        figInd = 0 
        for m in range(len(sampleMethods)):
            logging.debug(datasetNames[i] + " " + sampleMethods[m])

            outfileName = outputDir + datasetNames[i] + "GridResults.npz"
            data = numpy.load(outfileName)
            idealErrors = data["arr_0"]
            params = data["arr_1"]
            meanIdealErrorGrids = data["arr_2"]
            stdIdealErrorGrids = data["arr_3"]
            meanIdealPenGrids = data["arr_4"]
            stdIdealPenGrids = data["arr_5"]
            
            meanErrors = numpy.mean(idealErrors, 0)
            stdErrors = numpy.std(idealErrors, 0)  
            
            outfileName = outputDir + datasetNames[i] + sampleMethods[m] + "Results.npz"
            data = numpy.load(outfileName)
            meanErrorGrids = data["arr_2"]
            meanApproxGrids = data["arr_4"]

            #Print the errors
            for n in range(sampleSizes.shape[0]):
                j = 1
                plt.figure(figInd)
                plt.plot(numpy.log2(gammas), meanIdealErrorGrids[n, :], ".-", label="Accurate")
                j += 1    
                            
                sampleSizeInd = n
                methodInd = 0
                
                errorGrid = meanErrorGrids[sampleSizeInd, foldsInd, methodInd, :]
                plt.plot(numpy.log2(gammas), errorGrid, "--", label="CV")
                j += 1
                
                methodInd = 1
                errorGrid = meanErrorGrids[sampleSizeInd, foldsInd, methodInd, :]
                plt.plot(numpy.log2(gammas), errorGrid, label="PenVF+ ")
                j += 1
            
                for k in range(cvScalings.shape[0]):
                    methodInd = 2+k
                    errorGrid = meanErrorGrids[sampleSizeInd, foldsInd, methodInd, :]
                    plt.plot(numpy.log2(gammas), errorGrid, label="VFPen alpha=" + str(cvScalings[k]))
                    j += 1    
                
                plt.title("CV Grid n = " + str(sampleSizes[sampleSizeInd]) + " folds = " + str(foldsSet[foldsInd]))
                plt.legend()
                plt.xlabel("log(gamma)")
                plt.ylabel("error")
                
                figInd += 1    
    
                #Now plot ideal versus approx penalty 
                grid1 = meanIdealPenGrids[sampleSizeInd, :].flatten()
                grid2 = meanApproxGrids[sampleSizeInd, foldsInd, methodInd, :].flatten()
                           
                
                plt.figure(figInd)
                plt.scatter(grid1, grid2)
                plt.xlabel("Ideal penalty")
                plt.ylabel("Approximate penalty")
                plt.title("Penalty n = " + str(sampleSizes[sampleSizeInd]) + " folds = " + str(foldsSet[foldsInd]))
                
                figInd += 1  
    
        plt.show()

def plotGrids(datasetNames, sampleSizes, foldsSet, cvScalings, sampleMethods, fileNameSuffix, regression=False):
    for i in range(len(datasetNames)):
        figInd = 0 
        for m in range(len(sampleMethods)):
            logging.debug(datasetNames[i] + " " + sampleMethods[m])

            outfileName = outputDir + datasetNames[i] + "GridResults.npz"
            data = numpy.load(outfileName)
            idealErrors = data["arr_0"]
            params = data["arr_1"]
            meanIdealErrorGrids = data["arr_2"]
            stdIdealErrorGrids = data["arr_3"]
            meanIdealPenGrids = data["arr_4"]
            stdIdealPenGrids = data["arr_5"]
            
            meanErrors = numpy.mean(idealErrors, 0)
            stdErrors = numpy.std(idealErrors, 0)  
            
            outfileName = outputDir + datasetNames[i] + sampleMethods[m] + "Results.npz"
            data = numpy.load(outfileName)
            meanErrorGrids = data["arr_2"]
            stdErrorGrids = data["arr_3"]
            meanApproxGrids = data["arr_4"]

            #Print the errors
            for n in range(sampleSizes.shape[0]):                        
                j = 1
                plt.figure(figInd)
                plt.subplot(3, 3, j)
                plotErrorGrid(plt, Cs, gammas, meanIdealErrorGrids[n, :], "Accurate Grid", regression)
                j += 1

                sampleSizeInd = n
                methodInd = 0
                foldsInd = 1
                
                errorGrid = meanErrorGrids[sampleSizeInd, foldsInd, methodInd, :, :]
                plt.subplot(3, 3, j)
                plotErrorGrid(plt, Cs, gammas, errorGrid, "CV Grid n = " + str(sampleSizes[sampleSizeInd]) + " folds = " + str(foldsSet[foldsInd]), regression)
                j += 1
                
                methodInd = 1
                errorGrid = meanErrorGrids[sampleSizeInd, foldsInd, methodInd, :, :]
                plt.subplot(3, 3, j)
                plotErrorGrid(plt, Cs, gammas, errorGrid, "VFPen+ Grid n = " + str(sampleSizes[sampleSizeInd]) + " folds = " + str(foldsSet[foldsInd]), regression)
                j += 1

                for k in range(cvScalings.shape[0]):
                    methodInd = 2+k
                    errorGrid = meanErrorGrids[sampleSizeInd, foldsInd, methodInd, :, :]
                    plt.subplot(3, 3, j)
                    plotErrorGrid(plt, Cs, gammas, errorGrid, "VFPen Grid n = " + str(sampleSizes[sampleSizeInd]) + " folds = " + str(foldsSet[foldsInd]) + " Cv=" + str(cvScalings[k]), regression)
                    j += 1

                #Plot standard deviation of errors 
                j = 1
                plt.figure(figInd+1)        
                errorGrid = stdErrorGrids[sampleSizeInd, foldsInd, methodInd, :, :]
                plt.subplot(3, 3, j)
                plotErrorGrid(plt, Cs, gammas, errorGrid, "CV Grid n = " + str(sampleSizes[sampleSizeInd]) + " folds = " + str(foldsSet[foldsInd]), regression)
                j += 1

                methodInd = 1
                errorGrid = stdErrorGrids[sampleSizeInd, foldsInd, methodInd, :, :]
                plt.subplot(3, 3, j)
                plotErrorGrid(plt, Cs, gammas, errorGrid, "VFPen+ Grid n = " + str(sampleSizes[sampleSizeInd]) + " folds = " + str(foldsSet[foldsInd]), regression)
                j += 1

                for k in range(cvScalings.shape[0]):
                    methodInd = 2+k
                    errorGrid = stdErrorGrids[sampleSizeInd, foldsInd, methodInd, :, :]
                    plt.subplot(3, 3, j)
                    plotErrorGrid(plt, Cs, gammas, errorGrid, "VFPen Grid n = " + str(sampleSizes[sampleSizeInd]) + " folds = " + str(foldsSet[foldsInd]) + " Cv=" + str(cvScalings[k]), regression)
                    j += 1

                #Plot training error + ideal penality
                #methodInd = 1
                #errorGrid = meanErrorGrids[sampleSizeInd, foldsInd, methodInd, :]
                #trainErrorGrid = errorGrid - approxGrids[sampleSizeInd, foldsInd, methodInd, :]
                #idealErrorGrid = trainErrorGrid + idealGrids[sampleSizeInd, foldsInd, :]

                #plt.figure(figInd+1)
                #Fix for bug in benchmarkExp
                #idealErrorGrid = idealErrorGrid.mean(0)
                #plotErrorGrid(plt, Cs, gammas, idealErrorGrid, "Training error + ideal penality", regression)
                j += 1

                figInd += 2

            #Let's also plot the ideal versus approximate penalties


            sampleSizeInd = 0
            methodInd = 3
            foldsInd = 0

            for m in range(sampleSizes.shape[0]):
                sampleSizeInd = m
                print("Sample size = " + str(sampleSizes[m]))
                j = 1
                for k in range(foldsSet.shape[0]):
                    foldsInd = k
                    
                    plt.figure(m+figInd)
                    grid1 = meanIdealPenGrids[sampleSizeInd, :].flatten()
                    grid2 = meanApproxGrids[sampleSizeInd, foldsInd, methodInd, :].flatten()
                    
                    """
                    inds = numpy.where(meanApproxGrids[sampleSizeInd, foldsInd, methodInd, :] < 0.05) 
                    
                    for s, t in zip(inds[0], inds[1]): 
                        print(Cs[s], gammas[t])
                        
                    print(meanIdealErrorGrids[sampleSizeInd, :].shape)
                    print(Cs.shape)
                    plt.figure(100)
                    plt.plot(numpy.log(Cs), meanIdealErrorGrids[sampleSizeInd, 0, 0, :], label="ideal")
                    plt.plot(numpy.log(Cs), meanApproxGrids[sampleSizeInd, foldsInd, methodInd, 0, 0, :], label="approx")
                    plt.legend()
                    plt.show()
                    """
                    
                    plt.subplot(3, 2, j)
                    plt.scatter(grid1, grid2)
                    plt.xlabel("Ideal penalty")
                    plt.ylabel("Approximate penalty")
                    #plt.title("VFPen n=" + str(sampleSizes[sampleSizeInd]) + " V=" + str(foldsSet[foldsInd]) + " Cv=" + str(cvScalings[methodInd-2]))
                    xlims = plt.xlim()
                    plt.xlim([0, xlims[1]]) 
                    ylims = plt.ylim()
                    plt.ylim([0, ylims[1]]) 


                    plt.figure(m+sampleSizes.shape[0]+figInd)
                    grid1 = meanIdealErrorGrids[sampleSizeInd, :].min(1)
                    grid2 = meanApproxGrids[sampleSizeInd, foldsInd, methodInd, :].min(1)

                    error = numpy.abs(grid1-grid2)
                    
                    plt.subplot(2, 3, j)
                    plt.contourf(numpy.log2(Cs), numpy.log2(gammas), error, 100, antialiased=True)
                    plt.colorbar()
                    plt.xlabel("log C")
                    plt.ylabel("log gamma")
                    plt.title("VFPen n=" + str(sampleSizes[sampleSizeInd]) + " V=" + str(foldsSet[foldsInd]) + " Cv=" + str(cvScalings[methodInd-2]))

                    j += 1

            plt.show()

showCART = False   
showSVR = True 

if showSVR: 
    outputDir = PathDefaults.getOutputDir() + "modelPenalisation/regression/SVR/"
    datasetNames = ModelSelectUtils.getRegressionDatasets() 
    
    sampleSizes = numpy.array([50, 100, 200])
    foldsSet = numpy.arange(2, 13, 2)
    cvScalings = numpy.arange(0.6, 1.61, 0.2)
    numMethods = 2+cvScalings.shape[0]
    sampleMethods = ["CV"]
    fileNameSuffix = 'Results'    
    
    Cs = 2.0**numpy.arange(-10, 14, 2, dtype=numpy.float)
    gammas = 2.0**numpy.arange(-10, 4, 2, dtype=numpy.float)
    plotGrids(datasetNames, sampleSizes, foldsSet, cvScalings, sampleMethods, fileNameSuffix, True)

if showCART: 
    outputDir = PathDefaults.getOutputDir() + "modelPenalisation/regression/CART/"
    datasetNames = ModelSelectUtils.getRegressionDatasets() 
    
    sampleSizes = numpy.array([50, 100, 200])
    foldsSet = numpy.arange(2, 13, 2)
    cvScalings = numpy.arange(0.6, 1.61, 0.2)
    numMethods = 2+cvScalings.shape[0]
    sampleMethods = ["CV"]
    fileNameSuffix = 'Results'
    gammas = numpy.array(numpy.round(2**numpy.arange(1, 7.5, 0.5)-1), dtype=numpy.int)
    
    plotGridsCART(datasetNames, sampleSizes, foldsSet, cvScalings, sampleMethods, fileNameSuffix)
