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
        
        subgraphIndicesList = [range(0, self.startClusterSize)]
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
fileName = resultsDir + "ThreeClustErrors.npz"


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
 
k2s = [3, 6, 12, 24]
 
if saveResults: 
    numClusters = 3
    k1 = numClusters
    
    k3 = 90
    k4 = 90 
    T = 8 # index of iteration where exact decomposition is computed
    exactClusterer = IterativeSpectralClustering(k1, alg="exact")
    iascClusterers = []
    for k2 in k2s: 
        iascClusterers.append(IterativeSpectralClustering(k1, k2, alg="IASC", T=T)) 
    nystromClusterer = IterativeSpectralClustering(k1, k3=k3, alg="nystrom")
    ningsClusterer = NingSpectralClustering(k1, T=T)
    randSvdCluster = IterativeSpectralClustering(k1, k4=k4, alg="randomisedSvd")
    
    numRepetitions = 50
    #numRepetitions = 2
    do_Nings = True
    
    clustErrApprox = numpy.zeros((ps.shape[0], numGraphs, numRepetitions, len(k2s)))
    clustErrExact = numpy.zeros((ps.shape[0], numGraphs, numRepetitions))
    clustErrNings = numpy.zeros((ps.shape[0], numGraphs, numRepetitions))
    clustErrNystrom = numpy.zeros((ps.shape[0], numGraphs, numRepetitions))
    clustErrRandSvd = numpy.zeros((ps.shape[0], numGraphs, numRepetitions))
    
    for r in range(numRepetitions):
        Util.printIteration(r, 1, numRepetitions)
    
        for t in range(ps.shape[0]):
            logging.info("Run " + str(r) + "  p " + str(ps[t]))
            p = ps[t]
    
            logging.debug("Running exact method")
            graphIterator = ThreeClustIterator(p, numClusters, r).getIterator()
            clustListExact = exactClusterer.clusterFromIterator(graphIterator, False)
            
            logging.debug("Running approximate method")
            clustListApprox = []
            for i in range(len(k2s)): 
                graphIterator = ThreeClustIterator(p, numClusters, r).getIterator()
                clustListApprox.append(iascClusterers[i].clusterFromIterator(graphIterator, False)) 
            
            logging.debug("Running Nystrom method")
            graphIterator = ThreeClustIterator(p, numClusters, r).getIterator()
            clustListNystrom = nystromClusterer.clusterFromIterator(graphIterator, False)
    
            if do_Nings:
                logging.debug("Running Nings method")
                graphIterator = ThreeClustIterator(p, numClusters, r).getIterator()
                clustListNings = ningsClusterer.cluster(graphIterator)
                
            logging.debug("Running random SVD method")
            graphIterator = ThreeClustIterator(p, numClusters, r).getIterator()
            clustListRandSVD = randSvdCluster.clusterFromIterator(graphIterator, False)
    
            # computer rand index error for each iteration
            # error: proportion of pairs of vertices (x,y) s.t.
            #    (cl(x) == cl(y)) != (learned_cl(x) == learned_cl(y))
            for it in range(len(ThreeClustIterator().subgraphIndicesList)):
                  indicesList = ThreeClustIterator().subgraphIndicesList[it]
                  numUsedVertices = len(indicesList)
                  
                  for i in range(len(k2s)): 
                      clustErrApprox[t, it, r, i] += GraphUtils.randIndex(clustListApprox[i][it], indicesList)
                  clustErrExact[t, it, r] += GraphUtils.randIndex(clustListExact[it], indicesList)
                  clustErrNystrom[t, it, r] += GraphUtils.randIndex(clustListNystrom[it], indicesList)
                  if do_Nings:
                      clustErrNings[t, it, r] += GraphUtils.randIndex(clustListNings[it], indicesList)
                      
                  clustErrRandSvd[t, it, r] += GraphUtils.randIndex(clustListRandSVD[it], indicesList)
    
    numpy.savez(fileName, clustErrApprox, clustErrExact, clustErrNystrom, clustErrNings, clustErrRandSvd)
    logging.debug("Saved results as " + fileName)
else:  
    errors = numpy.load(fileName)
    clustErrApprox, clustErrExact, clustErrNystrom, clustErrNings, clustErrRandSvd = errors["arr_0"], errors["arr_1"], errors["arr_2"], errors["arr_3"]
    
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
    
    #Now lets plot the results
    iterations = numpy.arange(numGraphs)

    colourPlotStyles = ['k', 'r', 'g', 'b', 'y', 'm', 'c']
    linePlotStyles = ['-', '--', '-.', ':']
    pointPlotStyles = ['o', 'x', '+', '.']
    
    numMethods = 3+len(k2s)
    
    resultMeans = [] 
    resultStds = []
    names = []
    plotStyles = []
    for i in range(meanClustErrApprox.shape[2]): 
        resultMeans.append(meanClustErrApprox[:, :, i])
        resultStds.append(stdClustErrApprox[:, :, i])
        names.append("IASC " + str(k2s[i]))
        plotStyles.append(colourPlotStyles[0] + linePlotStyles[i])
    resultMeans.extend([meanClustErrExact, meanClustErrNings, meanClustErrNystrom, meanClustErrRandSvd])
    resultStds.extend([stdClustErrExact, stdClustErrNings, stdClustErrNystrom, stdClustErrRandSvd])
    names.extend(["Exact", "Ning", "Nystrom", "RandSVD"])
    for i in range(3): 
        plotStyles.append(colourPlotStyles[i+1] + linePlotStyles[0])
    
    plt.hold(True)
    for i_p in range(len(ps)):
        for i_res in range(numMethods):
            res = resultMeans[i_res]
        
            plt.figure(i_p)
            plt.plot(iterations, res[i_p, :], plotStyles[i_res], label=names[i_res])
            plt.ylim(0.33, 0.44)
            plt.grid(True)
            plt.xlabel("Graph no.")
            plt.ylabel("Rand Index")
            plt.legend(loc="upper left")
    plt.savefig(resultsDir + "ThreeClustErrors.eps")
    plt.show()

# to run
# python -c "execfile('exp/clusterexp/SyntheticExample.py')"