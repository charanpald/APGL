"""
Output some statistics for the citation iterator 

"""


import os
import sys
import logging
import numpy
from exp.clusterexp.CitationIterGenerator import CitationIterGenerator 
from apgl.graph import GraphStatistics 
from apgl.graph import SparseGraph, GraphUtils  
import matplotlib.pyplot as plt 
import scipy.sparse.linalg 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

plotCitation = True
plotBemol = False 
plotHIV = False 


maxGraphSize = 3000 
generator = CitationIterGenerator(maxGraphSize=maxGraphSize)
iterator = generator.getIterator()

subgraphIndicesList = []
for W in iterator: 
    subgraphIndicesList.append(range(W.shape[0])) 

#Try to find number of clusters at end of sequence by looking at eigengap 
k = 2000
L = GraphUtils.normalisedLaplacianSym(W)
logging.debug("Computing eigenvalues")
omega, Q = scipy.sparse.linalg.eigsh(L, min(k, L.shape[0]-1), which="SM", ncv = min(10*k, L.shape[0]))

omegaDiff = numpy.diff(omega)

print(omegaDiff)

plotInd = 0 
plt.figure(plotInd)
plt.plot(numpy.arange(omega.shape[0]), omega)
plt.xlabel("Eigenvalue index")
plt.ylabel("Eigenvalue")
plotInd += 1 

plt.figure(plotInd)
plt.plot(numpy.arange(omegaDiff.shape[0]), omegaDiff)
plt.xlabel("Eigenvalue index")
plt.ylabel("Eigenvalue diff")
plt.show()
#No obvious number of clusters and there are many edges 

    
"""
graph = SparseGraph(W.shape[0], W=W)


graphStats = GraphStatistics()
statsMatrix = graphStats.sequenceScalarStats(graph, subgraphIndicesList, slowStats=False)


plt.figure(0)
plt.plot(numpy.arange(statsMatrix.shape[0]), statsMatrix[:, graphStats.numVerticesIndex])
plt.xlabel("Graph index")
plt.ylabel("Num vertices")

plt.figure(1)
plt.plot(numpy.arange(statsMatrix.shape[0]), statsMatrix[:, graphStats.maxComponentSizeIndex])
plt.xlabel("Graph index")
plt.ylabel("Max component size")

plt.figure(2)
plt.plot(numpy.arange(statsMatrix.shape[0]), statsMatrix[:, graphStats.numComponentsIndex])
plt.xlabel("Graph index")
plt.ylabel("Num components")

plt.show()

"""