
import numpy
import logging
import sys
from apgl.util.ProfileUtils import ProfileUtils 
import pyximport
pyximport.install()
from exp.sandbox.predictors.TreeCriterion import findBestSplit
from apgl.data.ExamplesGenerator import ExamplesGenerator  



logging.basicConfig(stream=sys.stdout, level=logging.INFO)
numpy.random.seed(22)

class DecisionTreeLearnerProfile(object):
    def profileFindBestSplit(self):
        numExamples = 1000
        numFeatures = 100
        minSplit = 1
        maxDepth = 20
        
        generator = ExamplesGenerator()
        X, y = generator.generateBinaryExamples(numExamples, numFeatures)   
        
        nodeInds = numpy.arange(X.shape[0])
        argsortX = numpy.zeros(X.shape, numpy.int)      
        
        for i in range(X.shape[1]): 
            argsortX[:, i] = numpy.argsort(X[:, i])
            argsortX[:, i] = numpy.argsort(argsortX[:, i])            
        
        def run(): 
            for i in range(10): 
                findBestSplit(minSplit, X, y, nodeInds, argsortX) 
        
        ProfileUtils.profile('run()', globals(), locals())

profiler = DecisionTreeLearnerProfile()
profiler.profileFindBestSplit()
#0.685 
