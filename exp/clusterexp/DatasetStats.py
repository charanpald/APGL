"""
Output some statistics for the real graph datasets.  

"""
import os
import sys
import logging
import numpy
import itertools 
from exp.clusterexp.CitationIterGenerator import CitationIterGenerator 
from exp.clusterexp.HIVIterGenerator import HIVIterGenerator 
from apgl.graph import GraphStatistics 
from apgl.graph import SparseGraph, GraphUtils  
from apgl.util.PathDefaults import PathDefaults
from exp.clusterexp.BemolData import BemolData
import matplotlib.pyplot as plt 
import scipy.sparse.linalg 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

plotHIV = False 
plotCitation = False
plotBemol = True 

saveResults = True 
findEigs = False

if plotHIV: 
    def getIterator(): 
        generator = HIVIterGenerator()
        return generator.getIterator()
        
    resultsDir = PathDefaults.getOutputDir() + "cluster/HIV/Stats/"
    
if plotCitation: 
    
    def getIterator(): 
        maxGraphSize = 3000 
        generator = CitationIterGenerator(maxGraphSize=maxGraphSize)
        return generator.getIterator()
    
    resultsDir = PathDefaults.getOutputDir() + "cluster/Citation/Stats/"
if plotBemol: 
    def getIterator(): 
        dataDir = PathDefaults.getDataDir() + "cluster/"
        
        nbUser = 20000 # set to 'None' to have all users
        nbPurchasesPerIt = 100 # set to 'None' to take all the purchases per date
        startingIteration = 1000
        endingIteration = None # set to 'None' to have all iterations
        stepSize = 30    
        
        return itertools.islice(BemolData.getGraphIterator(dataDir, nbUser, nbPurchasesPerIt), startingIteration, endingIteration, stepSize)
    resultsDir = PathDefaults.getOutputDir() + "cluster/Bemol/Stats/"

if saveResults: 
    if not os.path.exists(resultsDir): 
       logging.warn("Directory did not exist: " + resultsDir + ", creating ...")
       os.mkdir(resultsDir)
       
    iterator = getIterator()
    
    subgraphIndicesList = []
    for W in iterator: 
        logging.debug("Graph size " + str(W.shape[0]))
        subgraphIndicesList.append(range(W.shape[0])) 
    
    #Try to find number of clusters at end of sequence by looking at eigengap 
    k = 2    
    
    if findEigs: 
        L = GraphUtils.normalisedLaplacianSym(W)
        
        logging.debug("Computing eigenvalues")
        omega, Q = scipy.sparse.linalg.eigsh(L, min(k, L.shape[0]-1), which="SM", ncv = min(20*k, L.shape[0]))
        
        omegaDiff = numpy.diff(omega)
    else: 
        omega = numpy.zeros(k)
        omegaDiff = numpy.zeros(k-1)
        
    #No obvious number of clusters and there are many edges 
    graph = SparseGraph(W.shape[0], W=W)
    
    logging.debug("Computing graph statistics")
    graphStats = GraphStatistics()
    statsMatrix = graphStats.sequenceScalarStats(graph, subgraphIndicesList, slowStats=False)
    
    numpy.savez(resultsDir + "GraphStats", statsMatrix, omega, omegaDiff)
    logging.debug("Saved results as " + resultsDir + "GraphStats.npz")
else:  
    arr = numpy.load(resultsDir + "GraphStats.npz")
    statsMatrix, omega, omegaDiff = arr["arr_0"], arr["arr_1"], arr["arr_2"]
    
    graphStats = GraphStatistics()
    plotInd = 0 
    
    if findEigs: 
        plt.figure(plotInd)
        plt.plot(numpy.arange(omega.shape[0]), omega)
        plt.xlabel("Eigenvalue index")
        plt.ylabel("Eigenvalue")
        plt.savefig(resultsDir + "Spectrum.eps")
        plotInd+=1 
        
        plt.figure(plotInd)
        plt.plot(numpy.arange(omegaDiff.shape[0]), omegaDiff)
        plt.xlabel("Eigenvalue index")
        plt.ylabel("Eigenvalue diff")
        plt.savefig(resultsDir + "SpectrumDiff.eps")
        plotInd +=1 
    
    plt.figure(plotInd)
    plt.plot(numpy.arange(statsMatrix.shape[0]), statsMatrix[:, graphStats.numVerticesIndex])
    plt.xlabel("Graph index")
    plt.ylabel("Num vertices")
    plt.savefig(resultsDir + "Vertices.eps")
    plotInd+=1 
    
    
    plt.figure(plotInd)
    plt.plot(numpy.arange(statsMatrix.shape[0]), statsMatrix[:, graphStats.numEdgesIndex])
    plt.xlabel("Graph index")
    plt.ylabel("Num edges")
    plt.savefig(resultsDir + "Edges.eps")
    plotInd+=1 
    
    plt.figure(plotInd)
    plt.plot(numpy.arange(statsMatrix.shape[0]), statsMatrix[:, graphStats.maxComponentSizeIndex])
    plt.xlabel("Graph index")
    plt.ylabel("Max component size")
    plotInd+=1 
    
    plt.figure(plotInd)
    plt.plot(numpy.arange(statsMatrix.shape[0]), statsMatrix[:, graphStats.numComponentsIndex])
    plt.xlabel("Graph index")
    plt.ylabel("Num components")
    plt.savefig(resultsDir + "Components.eps")
    plotInd+=1 
    
    plt.show()
