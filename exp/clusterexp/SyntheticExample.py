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
            W[self.endClusterSize*i:self.endClusterSize*(i+1), self.endClusterSize*i:self.endClusterSize*(i+1)] = pClust
            
        P = numpy.random.rand(self.numVertices, self.numVertices)
        W = numpy.array(P < W, numpy.float)
        upTriInds = numpy.triu_indices(self.numVertices)
        W[upTriInds] = 0
        W = W + W.T
        self.graph = SparseGraph(vList)
        self.graph.setWeightMatrix(W)
        
    def getIterator(self):
        return IncreasingSubgraphListIterator(self.graph, self.subgraphIndicesList)
  
numClusters = 3
k1 = numClusters
k2 = 20
k3 = 90
T = 8 # index of iteration where exact decomposition is computed
exactClusterer = IterativeSpectralClustering(k1, k2, alg="exact")
iascClusterer = IterativeSpectralClustering(k1, k2, alg="IASC", T=T)
nystromClusterer = IterativeSpectralClustering(k1, k3=k3, alg="nystrom")
ningsClusterer = NingSpectralClustering(k1, T=T)

#After 0.20 results are really bad 
ps = numpy.arange(0.05, 0.20, 0.1)
#ps = numpy.arange(0.05, 0.20, 0.1)
pClust = 0.3

perms = [l for l in itertools.permutations([0, 1, 2])]
#numRepetitions = 50
numRepetitions = 5
do_Nings = True
numGraphs = len(ThreeClustIterator().subgraphIndicesList) 

meanClustErrApprox = numpy.zeros((ps.shape[0], numGraphs, numRepetitions))
meanClustErrExact = numpy.zeros((ps.shape[0], numGraphs, numRepetitions))
meanClustErrNings = numpy.zeros((ps.shape[0], numGraphs, numRepetitions))
meanClustErrNystrom = numpy.zeros((ps.shape[0], numGraphs, numRepetitions))

for r in range(numRepetitions):
    Util.printIteration(r, 1, numRepetitions)
    clustErrApprox = numpy.zeros((numGraphs, ps.shape[0]))
    clustErrExact = numpy.zeros((numGraphs, ps.shape[0]))

    for t in range(ps.shape[0]):
        logging.info("Run " + str(r) + "  p " + str(ps[t]))
        p = ps[t]

        logging.debug("Running exact method")
        graphIterator = ThreeClustIterator(p, numClusters, r).getIterator()
        clustListExact = exactClusterer.clusterFromIterator(graphIterator, False)
        
        logging.debug("Running approximate method")
        graphIterator = ThreeClustIterator(p, numClusters, r).getIterator()
        clustListApprox = iascClusterer.clusterFromIterator(graphIterator, False)
        
        logging.debug("Running Nystrom method")
        graphIterator = ThreeClustIterator(p, numClusters, r).getIterator()
        clustListNystrom = nystromClusterer.clusterFromIterator(graphIterator, False)

        if do_Nings:
            logging.debug("Running Nings method")
            graphIterator = ThreeClustIterator(p, numClusters, r).getIterator()
            clustListNings = ningsClusterer.cluster(toDenseGraphListIterator(graphIterator))

        # computer rand index error for each iteration
        # error: proportion of pairs of vertices (x,y) s.t.
        #    (cl(x) == cl(y)) != (learned_cl(x) == learned_cl(y))
        for it in range(len(ThreeClustIterator().subgraphIndicesList)):
              indicesList = ThreeClustIterator().subgraphIndicesList[it]
              numUsedVertices = len(indicesList)

              meanClustErrExact[t, it, r] += GraphUtils.randIndex(clustListExact[it], indicesList)
              meanClustErrApprox[t, it, r] += GraphUtils.randIndex(clustListApprox[it], indicesList)
              meanClustErrNystrom[t, it, r] += GraphUtils.randIndex(clustListNystrom[it], indicesList)
              if do_Nings:
                  meanClustErrNings[t, it, r] += GraphUtils.randIndex(clustListNings[it], indicesList)

stdClustErrExact = meanClustErrExact.std(2)
stdClustErrApprox = meanClustErrApprox.std(2)
stdClustErrNystrom = meanClustErrNystrom.std(2)
stdClustErrNings = meanClustErrNings.std(2)

meanClustErrExact = meanClustErrExact.mean(2)
meanClustErrApprox = meanClustErrApprox.mean(2)
meanClustErrNystrom = meanClustErrNystrom.mean(2)
meanClustErrNings = meanClustErrNings.mean(2)

print(meanClustErrExact)
print(meanClustErrApprox)
print(meanClustErrNystrom)
print(meanClustErrNings)

print(stdClustErrExact)
print(stdClustErrApprox)
print(stdClustErrNystrom)
print(stdClustErrNings)

resultsDir = PathDefaults.getOutputDir() + "cluster/"
#Save results in a file
file_name = resultsDir + "ThreeClustErrors.dat"
try:
    res_file = open(file_name, 'w')
except(IOError), e:
    print "Warning: unable to open file '", file_name, "'\n", e
    print "=> results not saved"
else:
    comment = "# error for"
    for method in ["exact", "approx", "nings"]:
        for p in ps:
            comment = comment + " " + method + "_p" + str(p)
    res_file.write(comment + "\n")
    res = numpy.hstack((meanClustErrExact.T, meanClustErrApprox.T, meanClustErrNings.T))
    numpy.savetxt(res_file, res)
    
#Now lets plot the results
iterations = numpy.arange(numGraphs)
plotStyles = {}
plotStyles[0] = ['ko-', 'kx-', 'k+-', 'k.-', 'k*-', 'ks-']
plotStyles[1] = ['ko--', 'kx--', 'k+--', 'k.--', 'k*--', 'ks--']
plotStyles[2] = ['ko:', 'kx:', 'k+:', 'k:', 'k*:', 'ks:']

plt.hold(True)
for i_res in range(3):
    res = [meanClustErrExact, meanClustErrApprox, meanClustErrNystrom, meanClustErrNings][i_res]
    names = ["Exact", "IASC", "Nystrom", "Ning"]
    for i_p in range(len(ps)):
        plt.plot(iterations, res[i_p, :], plotStyles[i_res][i_p], label=names[i_res] + " p=" + str(ps[i_p]))
plt.xlabel("Graph index")
plt.ylabel("Rand Index")
plt.savefig(resultsDir + "ThreeClustErrors.eps")
plt.legend(loc="upper left")
plt.show()

# to run
# python -c "execfile('exp/clusterexp/SyntheticExample.py')"