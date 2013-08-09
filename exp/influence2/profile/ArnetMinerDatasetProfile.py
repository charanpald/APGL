import os 
import numpy
import logging
import sys
import igraph
import random  
from apgl.util.ProfileUtils import ProfileUtils
from exp.influence2.ArnetMinerDataset import ArnetMinerDataset 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class ArnetMinerDatasetProfile(object):
    def __init__(self):
        numpy.random.seed(21)        
        igraph._igraph.set_random_number_generator(random.WichmannHill(21))
        
    def profileFindAuthorsInField(self): 
        field = "Boosting"
        dataset = ArnetMinerDataset(field)
        
        
        ProfileUtils.profile('dataset.vectoriseDocuments()', globals(), locals())  
        
        
profiler = ArnetMinerDatasetProfile()
profiler.profileFindAuthorsInField() #211