import numpy
import logging
import sys
from apgl.util.ProfileUtils import ProfileUtils
from exp.recommendexp.NetflixDataset import NetflixDataset

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class NetflixDatasetProfile(object):
    def __init__(self):
        numpy.random.seed(21)        
        
        
    def profileTrainIterator(self):
        
        def run(): 
            dataset = NetflixDataset(maxIter=30)
    
            trainIterator = dataset.getTrainIteratorFunc()        
            
            for trainX in trainIterator: 
                print(trainX.shape)
            
        ProfileUtils.profile('run()', globals(), locals())    

profiler = NetflixDatasetProfile()
profiler.profileTrainIterator()
