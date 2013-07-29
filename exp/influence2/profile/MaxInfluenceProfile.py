import os 
import numpy
import logging
import sys
import igraph
import random  
from apgl.util.ProfileUtils import ProfileUtils
from exp.influence2.MaxInfluence import MaxInfluence 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class MaxInfluenceProfile(object):
    def __init__(self):
        numpy.random.seed(21)        
        igraph._igraph.set_random_number_generator(random.WichmannHill(21))
        
    def profileCelf(self):
         
        n = 100 
        p = 0.1
        graph = igraph.Graph.Erdos_Renyi(n, p)
        print(graph.summary())
            
        k = 5
        numpy.random.seed(21) 
        ProfileUtils.profile('MaxInfluence.celf(graph, k, p=0.5, numRuns=100)', globals(), locals())    

    def profileGreedyMethod(self):
         
        n = 100 
        p = 0.1
        graph = igraph.Graph.Erdos_Renyi(n, p)
        print(graph.summary())
            
        k = 5
        numpy.random.seed(21) 
        ProfileUtils.profile('MaxInfluence.greedyMethod(graph, k, p=0.5, numRuns=100)', globals(), locals())  

    def profileGreedyMethod2(self):
         
        n = 1000 
        p = 0.1
        graph = igraph.Graph.Erdos_Renyi(n, p)
        print(graph.summary())
            
        k = 5
        numpy.random.seed(21) 
        ProfileUtils.profile('MaxInfluence.greedyMethod2(graph, k, p=0.5, numRuns=100)', globals(), locals())  

    def profileSimulateCascades(self): 
        n = 500 
        p = 0.1
        graph = igraph.Graph.Erdos_Renyi(n, p)
            
        k = 50
        
        activeVertices = set(numpy.random.randint(0, n, 10))  
        numRuns = 100
        
        ProfileUtils.profile('MaxInfluence.simulateCascades(graph, activeVertices, numRuns, p=0.5)', globals(), locals())  

profiler = MaxInfluenceProfile()
profiler.profileGreedyMethod2()
#profiler.profileSimulateCascades()