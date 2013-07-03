import os 
import numpy
import logging
import sys
import igraph 
from apgl.util.ProfileUtils import ProfileUtils
from exp.influence2.MaxInfluence import MaxInfluence 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class MaxInfluenceProfile(object):
    def __init__(self):
        numpy.random.seed(21)        
        
        
    def profileCelf(self):
        n = 500 
        p = 0.1
        graph = igraph.Graph.Erdos_Renyi(n, p)
            
        k = 50
            
        ProfileUtils.profile('MaxInfluence.celf(graph, k, p=0.5)', globals(), locals())    

profiler = MaxInfluenceProfile()
profiler.profileCelf()
