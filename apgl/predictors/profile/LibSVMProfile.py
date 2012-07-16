import numpy
import logging
import sys
from apgl.util.ProfileUtils import ProfileUtils
from apgl.predictors.LibSVM import LibSVM
from apgl.util.Sampling import Sampling 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class LibSVMProfile(object):
    def __init__(self):
        self.folds = 10 
        self.paramDict = {} 
        self.paramDict["setC"] = 2**numpy.arange(-5, 5, dtype=numpy.float)  


    def profileModelSelect(self):
        learner = LibSVM()
        numExamples = 10000
        numFeatures = 10

        X = numpy.random.rand(numExamples, numFeatures)
        Y = numpy.array(numpy.random.rand(numExamples) < 0.1, numpy.int)*2-1

        def run():
            for i in range(5):
                print("Iteration " + str(i))
                idx = Sampling.crossValidation(self.folds, numExamples)
                learner.parallelModelSelect(X, Y, idx, self.paramDict)

        ProfileUtils.profile('run()', globals(), locals())

    def profileParallelPen(self): 
        learner = LibSVM(processes=8)
        learner.setChunkSize(2)
        numExamples = 10000
        numFeatures = 10

        X = numpy.random.rand(numExamples, numFeatures)
        Y = numpy.array(numpy.random.rand(numExamples) < 0.1, numpy.int)*2-1
        Cvs = [self.folds-1]

        def run():
            for i in range(2):
                print("Iteration " + str(i))
                idx = Sampling.crossValidation(self.folds, numExamples)
                learner.parallelPen(X, Y, idx, self.paramDict, Cvs)

        ProfileUtils.profile('run()', globals(), locals())

profiler = LibSVMProfile()
#profiler.profileModelSelect()
profiler.profileParallelPen() #42.6 