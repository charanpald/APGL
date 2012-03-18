
import numpy
import logging
import sys
from apgl.util.ProfileUtils import ProfileUtils 
from exp.sandbox.predictors.TreeCriterion import findBestSplit, findBestSplit2
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
        
        ProfileUtils.profile('findBestSplit2(minSplit, X, y) ', globals(), locals())

profiler = DecisionTreeLearnerProfile()
profiler.profileFindBestSplit()
#1.075
#0.475 after defining variables
#0.069 after static numpy arrays 
