"""
Observe the effect in the perturbations of Laplacians 
"""

import sys 
import logging
import numpy
import scipy 
import itertools 
import copy
import matplotlib.pyplot as plt 
from apgl.graph import *
from apgl.util.PathDefaults import PathDefaults
from exp.sandbox.IterativeSpectralClustering import IterativeSpectralClustering
from apgl.graph.GraphUtils import GraphUtils
from apgl.generator.SmallWorldGenerator import SmallWorldGenerator
from apgl.generator.ErdosRenyiGenerator import ErdosRenyiGenerator
from apgl.util.Util import Util 
from exp.clusterexp.BoundGraphIterator import BoundGraphIterator 

numpy.random.seed(21)
#numpy.seterr("raise")
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, linewidth=200, precision=3)
       
iterator = BoundGraphIterator()

for W in iterator: 
    L = GraphUtils.shiftLaplacian(W)
    u, V = numpy.linalg.eig(L.todense())
    u = numpy.flipud(numpy.sort(u))
    
    #print(u)

k1 = 3
k2 = 3
logging.debug("k=" + str(k1))

iterator = BoundGraphIterator()

clusterer = IterativeSpectralClustering(k1, k2)
clusterer.nb_iter_kmeans = 20
clusterer.computeBound = True 
logging.debug("Starting clustering")
clusterList, timeList, boundList = clusterer.clusterFromIterator(iterator, verbose=True, T=100)
boundList = numpy.array(boundList)
print(boundList)

errors = numpy.zeros(len(clusterList))

for i in range(len(clusterList)): 
    errors[i] = GraphUtils.randIndex(clusterList[i], iterator.realClustering)


plt.figure(0)
plt.plot(boundList[:, 0], boundList[:, 1], label="frobenius approx")
plt.plot(boundList[:, 0], boundList[:, 2], label="2-norm approx")
plt.plot(boundList[:, 0], boundList[:, 3], label="frobenius precise")
plt.plot(boundList[:, 0], boundList[:, 4], label="2-norm precise")
plt.xlabel("Graph no.")
plt.ylabel("||sin(theta)||")
plt.legend(loc="upper left")

#plt.figure(1)
#plt.plot(numpy.arange(errors.shape[0]), errors)
plt.show()



