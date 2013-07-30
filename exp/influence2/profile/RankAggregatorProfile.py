import os 
import numpy
import logging
import sys
import igraph
import random  
from apgl.util.ProfileUtils import ProfileUtils
from exp.influence2.RankAggregator import RankAggregator 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class RankAggregatorProfile(object):
    def __init__(self):
        numpy.random.seed(21)        
        igraph._igraph.set_random_number_generator(random.WichmannHill(21))
        
    def profileMC2(self): 
        numVals = 5000
        list1 = numpy.random.permutation(numVals).tolist()      
        list2 = numpy.random.permutation(numVals).tolist()   
        lists = [list1, list2]
        
        itemList = numpy.arange(numVals).tolist()
        
        ProfileUtils.profile('RankAggregator.MC2(lists, itemList)', globals(), locals())  
        

profiler = RankAggregatorProfile()
profiler.profileMC2()