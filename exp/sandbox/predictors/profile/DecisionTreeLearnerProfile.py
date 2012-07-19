
import numpy
import logging
import sys
from apgl.util.ProfileUtils import ProfileUtils 
from exp.sandbox.predictors.DecisionTreeLearner import DecisionTreeLearner
from apgl.data.ExamplesGenerator import ExamplesGenerator  
from sklearn.tree import DecisionTreeRegressor 
from apgl.util.Sampling import Sampling 

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
numpy.random.seed(22)

class DecisionTreeLearnerProfile(object):
    def profileLearnModel(self):
        numExamples = 1000
        numFeatures = 50
        minSplit = 10
        maxDepth = 20
        
        generator = ExamplesGenerator()
        X, y = generator.generateBinaryExamples(numExamples, numFeatures)   
        y = numpy.array(y, numpy.float)
            
        learner = DecisionTreeLearner(minSplit=minSplit, maxDepth=maxDepth, pruneType="REP-CV") 
        #learner.learnModel(X, y)
        #print("Done")
        ProfileUtils.profile('learner.learnModel(X, y) ', globals(), locals())
        
        print(learner.getTree().getNumVertices())

    def profileDecisionTreeRegressor(self): 
        numExamples = 1000
        numFeatures = 20
        minSplit = 10
        maxDepth = 20
        
        generator = ExamplesGenerator()
        X, y = generator.generateBinaryExamples(numExamples, numFeatures)   
            
        regressor = DecisionTreeRegressor(min_split=minSplit, max_depth=maxDepth, min_density=0.0)
        
        ProfileUtils.profile('regressor.fit(X, y)', globals(), locals())
        
    def profilePredict(self): 
        #Make the prdiction function faster 
        numExamples = 1000
        numFeatures = 20
        minSplit = 1
        maxDepth = 20
        
        generator = ExamplesGenerator()
        X, y = generator.generateBinaryExamples(numExamples, numFeatures)   
            
        learner = DecisionTreeLearner(minSplit=minSplit, maxDepth=maxDepth) 
        learner.learnModel(X, y)
        
        print(learner.getTree().getNumVertices())
        ProfileUtils.profile('learner.predict(X)', globals(), locals())
        
        print(learner.getTree().getNumVertices())

    def profileModelSelect(self):
        learner = DecisionTreeLearner(minSplit=5, maxDepth=30, pruneType="CART") 
        numExamples = 1000
        numFeatures = 10
        
        folds = 5
        
        paramDict = {} 
        paramDict["setGamma"] =  numpy.array(numpy.round(2**numpy.arange(1, 7.5, 0.5)-1), dtype=numpy.int)

        X = numpy.random.rand(numExamples, numFeatures)
        Y = numpy.array(numpy.random.rand(numExamples) < 0.1, numpy.int)*2-1

        def run():
            for i in range(5):
                print("Iteration " + str(i))
                idx = Sampling.crossValidation(folds, numExamples)
                learner.parallelModelSelect(X, Y, idx, paramDict)

        ProfileUtils.profile('run()', globals(), locals())
    
    def profileParallelPen(self):
        learner = DecisionTreeLearner(minSplit=5, maxDepth=30, pruneType="CART") 
        numExamples = 1000
        numFeatures = 10
        
        folds = 5
        
        paramDict = {} 
        paramDict["setGamma"] =  numpy.array(numpy.round(2**numpy.arange(1, 7.5, 0.5)-1), dtype=numpy.int)

        X = numpy.random.rand(numExamples, numFeatures)
        Y = numpy.array(numpy.random.rand(numExamples) < 0.1, numpy.int)*2-1
        Cvs = [folds-1]

        def run():
            for i in range(5):
                print("Iteration " + str(i))
                idx = Sampling.crossValidation(folds, numExamples)
                learner.parallelPen(X, Y, idx, paramDict, Cvs)

        ProfileUtils.profile('run()', globals(), locals())

profiler = DecisionTreeLearnerProfile()
#profiler.profileLearnModel() #0.418
#profiler.profileDecisionTreeRegressor() #0.020
#profiler.profilePredict() #0.024
profiler.profileParallelPen()
