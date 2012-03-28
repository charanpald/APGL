
import numpy
import logging
import sys
from apgl.util.ProfileUtils import ProfileUtils 
from exp.sandbox.predictors.DecisionTreeLearner import DecisionTreeLearner
from apgl.data.ExamplesGenerator import ExamplesGenerator  
from sklearn.tree import DecisionTreeRegressor 

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
numpy.random.seed(22)

class DecisionTreeLearnerProfile(object):
    def profileLearnModel(self):
        numExamples = 1000
        numFeatures = 20
        minSplit = 10
        maxDepth = 20
        
        generator = ExamplesGenerator()
        X, y = generator.generateBinaryExamples(numExamples, numFeatures)   
            
        learner = DecisionTreeLearner(minSplit=minSplit, maxDepth=maxDepth) 
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

profiler = DecisionTreeLearnerProfile()
#profiler.profileLearnModel() #0.418
#profiler.profileDecisionTreeRegressor() #0.020
profiler.profilePredict() #0.024
