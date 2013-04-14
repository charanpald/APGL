"""
Generate a synthetic sequence of graphs, and then cluster.
"""
import copy
import numpy
import logging
import sys
import itertools
import matplotlib.pyplot as plt
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Util import Util
from apgl.graph import SparseGraph, GraphUtils, GeneralVertexList
from apgl.generator import *
from exp.sandbox.IterativeSpectralClustering import IterativeSpectralClustering
from exp.sandbox.GraphIterators import IncreasingSubgraphListIterator, toDenseGraphListIterator
from exp.sandbox.NingSpectralClustering import NingSpectralClustering

numpy.random.seed(21)
numpy.set_printoptions(suppress=True, precision=3, linewidth=200, threshold=40000)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class ThreeClustIterator(object): 
    def __init__(self, p=0.1, numClusters=3, seed=21): 
        numpy.random.seed(seed)
        self.numClusters = numClusters
        self.startClusterSize = 20
        self.endClusterSize = 60
        self.clusterStep = 5
        self.pClust = 0.3
                
        self.numVertices = self.numClusters*self.endClusterSize
        vList = GeneralVertexList(self.numVertices)
        
        subgraphIndicesList = [list(range(0, self.startClusterSize))]
        subgraphIndicesList[0].extend(range(self.endClusterSize, self.endClusterSize+self.startClusterSize))
        subgraphIndicesList[0].extend(range(2*self.endClusterSize, 2*self.endClusterSize+self.startClusterSize))
        
        for i in range(self.startClusterSize, self.endClusterSize-self.clusterStep+1, self.clusterStep):
            subgraphIndices = copy.copy(subgraphIndicesList[-1])
            subgraphIndices.extend(range(i, i+self.clusterStep))
            subgraphIndices.extend(range(self.endClusterSize+i, self.endClusterSize+i+self.clusterStep))
            subgraphIndices.extend(range(2*self.endClusterSize+i, 2*self.endClusterSize+i+self.clusterStep))
            subgraphIndicesList.append(subgraphIndices)
        
        # to test removing
        # - increasing graph
        # do nothing
        # - decreasing graph
        #subgraphIndicesList.reverse()
        # - increasing and decreasing graph
        tmp = copy.copy(subgraphIndicesList[:-1])
        tmp.reverse()
        subgraphIndicesList.extend(tmp)
        self.subgraphIndicesList = subgraphIndicesList
        self.p = p 
        
        W = numpy.ones((self.numVertices, self.numVertices))*self.p
        
        for i in range(numClusters):
            W[self.endClusterSize*i:self.endClusterSize*(i+1), self.endClusterSize*i:self.endClusterSize*(i+1)] = self.pClust
            
        P = numpy.random.rand(self.numVertices, self.numVertices)
        W = numpy.array(P < W, numpy.float)
        upTriInds = numpy.triu_indices(self.numVertices)
        W[upTriInds] = 0
        W = W + W.T
        self.graph = SparseGraph(vList)
        self.graph.setWeightMatrix(W)
        
    def getIterator(self):
        return IncreasingSubgraphListIterator(self.graph, self.subgraphIndicesList)
  
#After 0.20 results are really bad 
ps = numpy.arange(0.1, 0.21, 0.1)
#ps = numpy.arange(0.05, 0.20, 0.1)  
numGraphs = len(ThreeClustIterator().subgraphIndicesList) 
saveResults = True 

resultsDir = PathDefaults.getOutputDir() + "cluster/"
fileNameError = resultsDir + "ThreeClustErrors.npz"
fileNameSinTheta = resultsDir + "ThreeClustSinTheta.npz"


#Plot spectrums of largest graphs 
"""  
iterator = ThreeClustIterator(0.1).getIterator()
for W in iterator: 
    if W.shape[0] == 180: 
        L = GraphUtils.shiftLaplacian(W)
        u, V = numpy.linalg.eig(L.todense())
        u = numpy.flipud(numpy.sort(u))
        
        print(u)
        plt.plot(numpy.arange(u.shape[0]), u)

iterator = ThreeClustIterator(0.2).getIterator()
plt.figure(10)
for W in iterator: 
    if W.shape[0] == 180: 
        L = GraphUtils.shiftLaplacian(W)
        u, V = numpy.linalg.eig(L.todense())
        u = numpy.flipud(numpy.sort(u))
        
        print(u)
        plt.plot(numpy.arange(u.shape[0]), u)

plt.show()
"""
 
k2s = [3, 6, 12, 24, 150]
k3s = [3, 24, 90]
k4s = [3, 24]
# debug of IASC
#k2s = [3, 6, 12, 24, 150]
#k3s = [3]
#k4s = [3]
 
if saveResults: 
    numClusters = 3
    k1 = numClusters
    
    T = 8 # index of iteration where exact decomposition is computed
    exactClusterer = IterativeSpectralClustering(k1, alg="exact", computeSinTheta=True)
    iascClusterers = []
    for k2 in k2s: 
        iascClusterers.append(IterativeSpectralClustering(k1, k2, alg="IASC", computeSinTheta=True, T=T)) 
    nystromClusterers = []
    for k3 in k3s: 
        nystromClusterers.append(IterativeSpectralClustering(k1, k3=k3, alg="nystrom", computeSinTheta=True))
    ningsClusterer = NingSpectralClustering(k1, T=T, computeSinTheta=True)
    randSvdClusterers = []
    for k4 in k4s: 
        randSvdClusterers.append(IterativeSpectralClustering(k1, k4=k4, alg="randomisedSvd", computeSinTheta=True))
    
    numRepetitions = 50
#    numRepetitions = 2
    do_Nings = True
    
    clustErrApprox = numpy.zeros((ps.shape[0], numGraphs, numRepetitions, len(k2s)))
    clustErrExact = numpy.zeros((ps.shape[0], numGraphs, numRepetitions))
    clustErrNings = numpy.zeros((ps.shape[0], numGraphs, numRepetitions))
    clustErrNystrom = numpy.zeros((ps.shape[0], numGraphs, numRepetitions, len(k3s)))
    clustErrRandSvd = numpy.zeros((ps.shape[0], numGraphs, numRepetitions, len(k4s)))
    sinThetaApprox = numpy.zeros((ps.shape[0], numGraphs, numRepetitions, len(k2s)))
    sinThetaExact = numpy.zeros((ps.shape[0], numGraphs, numRepetitions))
    sinThetaNings = numpy.zeros((ps.shape[0], numGraphs, numRepetitions))
    sinThetaNystrom = numpy.zeros((ps.shape[0], numGraphs, numRepetitions, len(k3s)))
    sinThetaRandSvd = numpy.zeros((ps.shape[0], numGraphs, numRepetitions, len(k4s)))
    
    for r in range(numRepetitions):
        Util.printIteration(r, 1, numRepetitions)
    
        for t in range(ps.shape[0]):
            logging.info("Run " + str(r) + "  p " + str(ps[t]))
            p = ps[t]
    
            logging.debug("Running exact method")
            graphIterator = ThreeClustIterator(p, numClusters, r).getIterator()
            resExact = exactClusterer.clusterFromIterator(graphIterator, True)
            
            logging.debug("Running approximate method")
            resApproxList = []
            for i in range(len(k2s)): 
                graphIterator = ThreeClustIterator(p, numClusters, r).getIterator()
                resApproxList.append(iascClusterers[i].clusterFromIterator(graphIterator, True)) 
            
            logging.debug("Running Nystrom method")
            resNystromList = []
            for i in range(len(k3s)): 
                graphIterator = ThreeClustIterator(p, numClusters, r).getIterator()
                resNystromList.append(nystromClusterers[i].clusterFromIterator(graphIterator, True))
    
            if do_Nings:
                logging.debug("Running Nings method")
                graphIterator = ThreeClustIterator(p, numClusters, r).getIterator()
                resNings = ningsClusterer.cluster(graphIterator, True)
                
            logging.debug("Running random SVD method")
            resRandSVDList = []
            for i in range(len(k4s)): 
                graphIterator = ThreeClustIterator(p, numClusters, r).getIterator()
                resRandSVDList.append(randSvdClusterers[i].clusterFromIterator(graphIterator, True))
    
            # computer rand index error for each iteration
            # error: proportion of pairs of vertices (x,y) s.t.
            #    (cl(x) == cl(y)) != (learned_cl(x) == learned_cl(y))
            for it in range(len(ThreeClustIterator().subgraphIndicesList)):
                  indicesList = ThreeClustIterator().subgraphIndicesList[it]
                  numUsedVertices = len(indicesList)
                  
                  for k in range(len(k2s)): 
                      clustErrApprox[t, it, r, k] = GraphUtils.randIndex(resApproxList[k][0][it], indicesList)
                  clustErrExact[t, it, r] = GraphUtils.randIndex(resExact[0][it], indicesList)
                  for k in range(len(k3s)): 
                      clustErrNystrom[t, it, r, k] = GraphUtils.randIndex(resNystromList[k][0][it], indicesList)
                  if do_Nings:
                      clustErrNings[t, it, r] = GraphUtils.randIndex(resNings[0][it], indicesList)
                  for k in range(len(k4s)): 
                      clustErrRandSvd[t, it, r, k] = GraphUtils.randIndex(resRandSVDList[k][0][it], indicesList)
    
            # store sin(Theta)
            for k in range(len(k2s)): 
                sinThetaApprox[t, :, r, k] = resApproxList[k][2]["sinThetaList"]
            sinThetaExact[t, :, r] = resExact[2]["sinThetaList"]
            for k in range(len(k3s)): 
                sinThetaNystrom[t, :, r, k] = resNystromList[k][2]["sinThetaList"]
            if do_Nings:
                sinThetaNings[t, :, r] = resNings[2]["sinThetaList"]
            for k in range(len(k4s)): 
                sinThetaRandSvd[t, :, r, k] = resRandSVDList[k][2]["sinThetaList"]
    
    numpy.savez(fileNameError, clustErrApprox, clustErrExact, clustErrNystrom, clustErrNings, clustErrRandSvd)
    logging.debug("Saved results as " + fileNameError)
    numpy.savez(fileNameSinTheta, sinThetaApprox, sinThetaExact, sinThetaNystrom, sinThetaNings, sinThetaRandSvd)
    logging.debug("Saved results as " + fileNameSinTheta)
else:  
    errors = numpy.load(fileNameError)
    clustErrApprox, clustErrExact, clustErrNystrom, clustErrNings, clustErrRandSvd = errors["arr_0"], errors["arr_1"], errors["arr_2"], errors["arr_3"], errors["arr_4"]
    sinTheta = numpy.load(fileNameSinTheta)
    sinThetaApprox, sinThetaExact, sinThetaNystrom, sinThetaNings, sinThetaRandSvd = sinTheta["arr_0"], sinTheta["arr_1"], sinTheta["arr_2"], sinTheta["arr_3"], sinTheta["arr_4"]
    
    meanClustErrExact = clustErrExact.mean(2)
    meanClustErrApprox = clustErrApprox.mean(2)
    meanClustErrNystrom = clustErrNystrom.mean(2)
    meanClustErrNings = clustErrNings.mean(2)
    meanClustErrRandSvd = clustErrRandSvd.mean(2)
    
    stdClustErrExact = clustErrExact.std(2)
    stdClustErrApprox = clustErrApprox.std(2)
    stdClustErrNystrom = clustErrNystrom.std(2)
    stdClustErrNings = clustErrNings.std(2)
    stdClustErrRandSvd = clustErrRandSvd.std(2)
    
    meanSinThetaExact = sinThetaExact.mean(2)
    meanSinThetaApprox = sinThetaApprox.mean(2)
    meanSinThetaNystrom = sinThetaNystrom.mean(2)
    meanSinThetaNings = sinThetaNings.mean(2)
    meanSinThetaRandSvd = sinThetaRandSvd.mean(2)
    
    stdSinThetaExact = sinThetaExact.std(2)
    stdSinThetaApprox = sinThetaApprox.std(2)
    stdSinThetaNystrom = sinThetaNystrom.std(2)
    stdSinThetaNings = sinThetaNings.std(2)
    stdSinThetaRandSvd = sinThetaRandSvd.std(2)
    
    print(meanClustErrExact)
    print(meanClustErrApprox)
    print(meanClustErrNystrom)
    print(meanClustErrNings)
    print(meanClustErrRandSvd)
    
    print(stdClustErrExact)
    print(stdClustErrApprox)
    print(stdClustErrNystrom)
    print(stdClustErrNings)
    print(stdClustErrRandSvd)
    
    print("<sin(Theta)>")
    print(meanSinThetaExact)
    print(meanSinThetaApprox)
    print(meanSinThetaNystrom)
    print(meanSinThetaNings)
    print(meanSinThetaRandSvd)
    print("</sin(Theta)>")
    
    #Now lets plot the results
    iterations = numpy.arange(numGraphs)

    colourPlotStyles = ['k', 'r', 'g', 'b', 'y', 'm', 'c']
    linePlotStyles = ['-', '--', '-.', ':', ':']
    pointPlotStyles = ['o', 'x', '+', '.', '*']
    
    resultMeans = {"error": [], "sinTheta": []}
    resultStds = {"error": [], "sinTheta": []}
    names = []
    plotStyles = []
    usedK2Inds = [0,1,2,3]
    for i_usedK2, i_k2 in enumerate(usedK2Inds): 
        resultMeans["error"].append(meanClustErrApprox[:, :, i_k2])
        resultStds["error"].append(stdClustErrApprox[:, :, i_k2])
        resultMeans["sinTheta"].append(meanSinThetaApprox[:, :, i_k2])
        resultStds["sinTheta"].append(stdSinThetaApprox[:, :, i_k2])
        names.append("IASC " + str(k2s[i_k2]))
        plotStyles.append(colourPlotStyles[0] + linePlotStyles[i_usedK2])
    resultMeans["error"].append(meanClustErrExact)
    resultStds["error"].append(stdClustErrExact)
    resultMeans["sinTheta"].append(meanSinThetaExact)
    resultStds["sinTheta"].append(stdSinThetaExact)
    names.append("Exact")
    plotStyles.append(colourPlotStyles[1] + linePlotStyles[0])
    resultMeans["error"].append(meanClustErrNings)
    resultStds["error"].append(stdClustErrNings)
    resultMeans["sinTheta"].append(meanSinThetaNings)
    resultStds["sinTheta"].append(stdSinThetaNings)
    names.append("Ning")
    plotStyles.append(colourPlotStyles[2] + linePlotStyles[0])
    usedK3Inds = [2]
    for i_usedK3, i_k3 in enumerate(usedK3Inds): 
        resultMeans["error"].append(meanClustErrNystrom[:, :, i_k3])
        resultStds["error"].append(stdClustErrNystrom[:, :, i_k3])
        resultMeans["sinTheta"].append(meanSinThetaNystrom[:, :, i_k3])
        resultStds["sinTheta"].append(stdSinThetaNystrom[:, :, i_k3])
        names.append("Nyst " + str(k3s[i_k3]))
        plotStyles.append(colourPlotStyles[3] + linePlotStyles[i_usedK3])
    usedK4Inds = [1]
    for i_usedK4, i_k4 in enumerate(usedK4Inds): 
        resultMeans["error"].append(meanClustErrRandSvd[:, :, i_k4])
        resultStds["error"].append(stdClustErrRandSvd[:, :, i_k4])
        resultMeans["sinTheta"].append(meanSinThetaRandSvd[:, :, i_k4])
        resultStds["sinTheta"].append(stdSinThetaRandSvd[:, :, i_k4])
        names.append("RSVD " + str(k4s[i_k4]))
        plotStyles.append(colourPlotStyles[4] + linePlotStyles[i_usedK4])
    
    for i_p in range(len(ps)):
        for i_res in range(len(names)):
            res = resultMeans["error"][i_res]
        
            plt.figure(i_p)
            plt.plot(iterations, res[i_p, :], plotStyles[i_res], label=names[i_res])
            plt.ylim(0.33, 0.44)
            plt.grid(True)
            plt.xlabel("Graph no.")
            plt.ylabel("Rand Index")
            plt.legend(loc="upper left")
        plt.savefig(resultsDir + "ThreeClustErrors_p" + str(ps[i_p]) + ".eps")
        logging.info(resultsDir + "ThreeClustErrors_p" + str(ps[i_p]) + ".eps")


    for i_p in range(len(ps)):
        for i_res in range(len(names)):
            if i_res != len(usedK2Inds): # do not print exact results
                res = resultMeans["sinTheta"][i_res]
            
                plt.figure(len(ps)+i_p)
                plt.plot(iterations, res[i_p, :], plotStyles[i_res], label=names[i_res])
                plt.ylim(0, 2)
                plt.grid(True)
                plt.xlabel("Graph no.")
                plt.ylabel("||sin(Theta)||")
                plt.legend(loc="upper left", ncol=2)
        plt.savefig(resultsDir + "ThreeClustSinThetas_p" + str(ps[i_p]) + ".eps")
        logging.info(resultsDir + "ThreeClustSinThetas_p" + str(ps[i_p]) + ".eps")
plt.show()

# to run
# python -c "execfile('exp/clusterexp/SyntheticExample.py')"