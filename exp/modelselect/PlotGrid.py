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
    
    error10 = numpy.array([ (numpy.log2(Cs[ii]),numpy.log2(gammas[jj]),errorGrid[jj,ii].min()) for ii in range(Cs.shape[0]) for jj in range(gammas.shape[0]) ])
    top10 = numpy.array(sorted(error10,key=itemgetter(2)))[0:10, :]
    plt.scatter(top10[:, 0], top10[:, 1])

def plotErrorGridCART(plt, gammas, errorGrid, title):    
    plt.plot(numpy.log2(gammas), errorGrid, label=title)
    plt.xlabel("log(gamma)")
    plt.ylabel("error")
    plt.title(title)
    
def plotGridsCART(datasetNames, sampleSizes, foldsSet, cvScalings, sampleMethods, fileNameSuffix, regression=False): 
    figInd = 0 
    for i in range(len(datasetNames)):
        figInd = 0 
        for m in range(len(sampleMethods)):
            logging.debug(datasetNames[i] + " " + sampleMethods[m])

            #Print the errors
            for n in range(sampleSizes.shape[0]):
    
                outfileName = outputDir + datasetNames[i] + "GridResults" + str(sampleSizes[n]) + ".npz"
                data = numpy.load(outfileName)
                errors = data["arr_0"]
                meanErrors = numpy.mean(errors, 0)
                stdErrors = numpy.std(errors, 0)     
                
                
                j = 1
                plt.figure(figInd)
                #plt.subplot(3, 3, j)
                plt.plot(numpy.log2(gammas), meanErrors, ".-", label="Accurate")
                j += 1    
                
                outfileName = outputDir + datasetNames[i] + sampleMethods[m] + fileNameSuffix  + ".npz"
                data = numpy.load(outfileName)
                meanErrorGrids = data["arr_2"]
                stdErrorGrids= data["arr_3"]
                idealGrids = data["arr_4"]
                stdIdealGrids = data["arr_5"]
                approxGrids = data["arr_6"]
                stdApproxGrids = data["arr_7"]
            
                sampleSizeInd = n
                methodInd = 0
                foldsInd = 1
                
                errorGrid = meanErrorGrids[sampleSizeInd, foldsInd, methodInd, :]
                #plt.subplot(3, 3, j)
                plt.plot(numpy.log2(gammas), errorGrid, "--", label="CV")
                j += 1
                
                methodInd = 1
                errorGrid = meanErrorGrids[sampleSizeInd, foldsInd, methodInd, :]
                #plt.subplot(3, 3, j)
                plt.plot(numpy.log2(gammas), errorGrid, label="BIC")
                j += 1
            
                for k in range(cvScalings.shape[0]):
                    methodInd = 2+k
                    errorGrid = meanErrorGrids[sampleSizeInd, foldsInd, methodInd, :]
                    #plt.subplot(3, 3, j)
                    plt.plot(numpy.log2(gammas), errorGrid, label="VFPen Cv=" + str(cvScalings[k]))
                    j += 1    
                
                plt.title("CV Grid n = " + str(sampleSizes[sampleSizeInd]) + " folds = " + str(foldsSet[foldsInd]))
                plt.legend()
                plt.xlabel("log(gamma)")
                plt.ylabel("error")
                
                figInd += 1    
    
                #Now plot ideal versus approx penalty 
                grid1 = idealGrids[sampleSizeInd, foldsInd, methodInd, ].flatten()
                grid2 = approxGrids[sampleSizeInd, foldsInd, methodInd, :].flatten()
                
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

            #Print the errors
            for n in range(sampleSizes.shape[0]):
                outfileName = outputDir + datasetNames[i] + "GridResults" + str(sampleSizes[n]) + ".npz"
                data = numpy.load(outfileName)
                errors = data["arr_0"]
                meanErrors = numpy.mean(errors, 0)
                stdErrors = numpy.std(errors, 0)                
                
                j = 1
                plt.figure(figInd)
                plt.subplot(3, 3, j)
                plotErrorGrid(plt, Cs, gammas, meanErrors, "Accurate Grid", regression)
                j += 1
                
                

                outfileName = outputDir + datasetNames[i] + sampleMethods[m] + fileNameSuffix  + ".npz"
                data = numpy.load(outfileName)
                meanErrorGrids = data["arr_2"]
                stdErrorGrids= data["arr_3"]
                idealGrids = data["arr_4"]
                stdIdealGrids = data["arr_5"]
                approxGrids = data["arr_6"]
                stdApproxGrids = data["arr_7"]

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
                plotErrorGrid(plt, Cs, gammas, errorGrid, "BIC Grid n = " + str(sampleSizes[sampleSizeInd]) + " folds = " + str(foldsSet[foldsInd]), regression)
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
                plotErrorGrid(plt, Cs, gammas, errorGrid, "BIC Grid n = " + str(sampleSizes[sampleSizeInd]) + " folds = " + str(foldsSet[foldsInd]), regression)
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
            #BIC penalisation
            methodInd = 3
            foldsInd = 0

            #Fix for bug in benchmarkExp
            idealGrids = idealGrids.mean(2)            
            
            grid1 = idealGrids[sampleSizeInd, foldsInd, :, :].flatten()
            grid2 = approxGrids[sampleSizeInd, foldsInd, methodInd, :, :].flatten()



            for m in range(sampleSizes.shape[0]):
                sampleSizeInd = m
                print("Sample size = " + str(sampleSizes[m]))
                j = 1
                for k in range(foldsSet.shape[0]):
                    foldsInd = k
                    
                    plt.figure(m+figInd)
                    grid1 = idealGrids[sampleSizeInd, foldsInd, :].flatten()
                    grid2 = approxGrids[sampleSizeInd, foldsInd, methodInd, :].flatten()
                    
                    plt.subplot(3, 2, j)
                    plt.scatter(grid1, grid2)
                    plt.xlabel("Ideal penalty")
                    plt.ylabel("Approximate penalty")
                    #plt.title("VFPen n=" + str(sampleSizes[sampleSizeInd]) + " V=" + str(foldsSet[foldsInd]) + " Cv=" + str(cvScalings[methodInd-2]))
                    xlims = plt.xlim()
                    plt.xlim([0, xlims[1]]) 
                    ylims = plt.ylim()
                    plt.ylim([0, ylims[1]]) 

                    lineCoeff, residuals, rank, singular_values, rcond = numpy.polyfit(grid1, grid2, 1, full=True)  
                    print(lineCoeff)
                    print(residuals)
                    plt.plot(numpy.array([0, numpy.max(grid1)]), numpy.array([lineCoeff[1], lineCoeff[0]*numpy.max(grid1) ]))

                    plt.figure(m+sampleSizes.shape[0]+figInd)
                    grid1 = idealGrids[sampleSizeInd, foldsInd, :, :]
                    grid2 = approxGrids[sampleSizeInd, foldsInd, methodInd, :, :]

                    error = numpy.abs(grid1-grid2)
                    error = error.min(1)
                    
                    plt.subplot(2, 3, j)
                    plt.contourf(numpy.log2(Cs), numpy.log2(gammas), error, 100, antialiased=True)
                    plt.colorbar()
                    plt.xlabel("log C")
                    plt.ylabel("log gamma")
                    plt.title("VFPen n=" + str(sampleSizes[sampleSizeInd]) + " V=" + str(foldsSet[foldsInd]) + " Cv=" + str(cvScalings[methodInd-2]))

                    #This is the sum of the standard deviations of the ideal and approx grids 
                    """
                    plt.figure(m+sampleSizes.shape[0]*2+figInd)
                    grid1 = idealGrids[sampleSizeInd, foldsInd, :, :]
                    grid2 = approxGrids[sampleSizeInd, foldsInd, methodInd, :, :]
                    plt.subplot(2, 3, j)
                    plt.contourf(numpy.log2(gammas), numpy.log2(Cs), stdIdealGrids[sampleSizeInd, foldsInd, :, :] + stdApproxGrids[sampleSizeInd, foldsInd, methodInd, :, :], 100, antialiased=True)
                    plt.colorbar()
                    plt.xlabel("Gamma")
                    plt.ylabel("C")
                    plt.title("VFPen n=" + str(sampleSizes[sampleSizeInd]) + " V=" + str(foldsSet[foldsInd]) + " Cv=" + str(cvScalings[methodInd]))
                    """                    
                    
                    j += 1

            plt.show()

#outputDir = PathDefaults.getOutputDir() + "modelPenalisation/classification/"
#outputDir = PathDefaults.getOutputDir() + "modelPenalisation/regression/SVR/"
outputDir = PathDefaults.getOutputDir() + "modelPenalisation/regression/CART/"


sampleSizes = numpy.array([50, 100, 200])
foldsSet = numpy.arange(2, 13, 2)
cvScalings = numpy.arange(0.6, 1.61, 0.2)
numMethods = 2+cvScalings.shape[0]
#sampleMethods = ["CV","SS", "SS66", "SS90"]
sampleMethods = ["CV"]
fileNameSuffix = 'Results'

#datasetNames = ModelSelectUtils.getRatschDatasets()
datasetNames = ModelSelectUtils.getRegressionDatasets() 
#datasetNames = ["add10"] 
#datasetNames = ["slice-loc"]
#datasetNames = ["abalone"]
#datasetNames = ["comp-activ"]

#Cs = 2.0**numpy.arange(-10, 14, 2, dtype=numpy.float)
#gammas = 2.0**numpy.arange(-10, 4, 2, dtype=numpy.float)
#plotGrids(datasetNames, sampleSizes, foldsSet, cvScalings, sampleMethods, fileNameSuffix, True)

gammas = numpy.array(numpy.round(2**numpy.arange(1, 10, 0.5)-1), dtype=numpy.int)
plotGridsCART(datasetNames, sampleSizes, foldsSet, cvScalings, sampleMethods, fileNameSuffix)
