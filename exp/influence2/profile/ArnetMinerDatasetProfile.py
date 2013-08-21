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
        
    def profileVectoriseDocuments(self): 
        field = "Boosting"
        dataset = ArnetMinerDataset(field)
        
        
        ProfileUtils.profile('dataset.vectoriseDocuments()', globals(), locals())  
  
    def profileComputeLDA(self): 
        field = "Boosting"
        dataset = ArnetMinerDataset(field)
        dataset.overwrite = True
        dataset.overwriteVectoriser = True
        dataset.overwriteModel = True
        dataset.maxRelevantAuthors = 100
        dataset.k = 200
        dataset.dataFilename = dataset.dataDir + "DBLP-citation-100000.txt"
        
        ProfileUtils.profile('dataset.computeLDA()', globals(), locals()) 
        
    def profileModelSelection(self): 
        dataset = ArnetMinerDataset(runLSI=False)   
        dataset.overwrite = True
        dataset.overwriteVectoriser = True
        dataset.overwriteModel = True
        
        dataset.dataFilename = dataset.dataDir + "DBLP-citation-100000.txt"
        
        ProfileUtils.profile('dataset.modelSelection()', globals(), locals())
        
        
profiler = ArnetMinerDatasetProfile()
#profiler.profileVectoriseDocuments() #211
profiler.profileModelSelection() # 121 