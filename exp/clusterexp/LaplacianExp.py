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
       
k1 = 3
k2 = 3
logging.debug("k=" + str(k1))
numRepetitions = 50
numGraphs = 80

saveResults = False 
resultsDir = PathDefaults.getOutputDir() + "cluster/"
fileName = resultsDir + "ErrorBoundTheorem44.npy"

if saveResults: 
    errors = numpy.zeros((numGraphs, numRepetitions))  
    allBoundLists = numpy.zeros((numRepetitions, numGraphs-1, 5))
    
    for r in range(numRepetitions): 
        iterator = BoundGraphIterator(numGraphs=numGraphs)
        
        clusterer = IterativeSpectralClustering(k1, k2, T=100, computeBound=True, alg="IASC")
        clusterer.nb_iter_kmeans = 20
        logging.debug("Starting clustering")
        clusterList, timeList, boundList = clusterer.clusterFromIterator(iterator, verbose=True)
        allBoundLists[r, :, :] = numpy.array(boundList)
        
        
        for i in range(len(clusterList)): 
            errors[i, r] = GraphUtils.randIndex(clusterList[i], iterator.realClustering)
            
    print(allBoundLists.mean(0))
    
    numpy.save(fileName, allBoundLists)
    logging.debug("Saved results as " + fileName)
else: 
    allBoundLists = numpy.load(fileName) 
    boundList = allBoundLists.mean(0)
    stdBoundList = allBoundLists.std(0)
    stdBoundList[:, 0] = boundList[:, 0]
    
    plotStyles1 = ['k-', 'k--', 'k-.', 'k:', 'b--', 'b-.', 'g-', 'g--', 'g-.', 'r-', 'r--', 'r-.']    
    print(boundList)
    print(stdBoundList)

    plt.figure(0)
    plt.plot(boundList[:, 0], boundList[:, 1], plotStyles1[0], label="Frobenius approx")
    plt.plot(boundList[:, 0], boundList[:, 2], plotStyles1[1], label="2-norm approx")
    plt.plot(boundList[:, 0], boundList[:, 3], plotStyles1[2], label="Frobenius precise")
    plt.plot(boundList[:, 0], boundList[:, 4], plotStyles1[3], label="2-norm precise")
    plt.xlabel("Graph no.")
    plt.ylabel("||sin(theta)||")
    plt.legend(loc="upper left")
    plt.grid(True)
    
    #plt.figure(1)
    #plt.plot(numpy.arange(errors.shape[0]), errors)
    plt.show()



