'''
Created on 16 Jul 2009

@author: charanpal

Classify a point according to the conditional probability P(y | X) = prod_i P(y | X_i), 
assuming that X_i's are independent. 
'''

from apgl.predictors.AbstractPredictor import AbstractPredictor
from apgl.util.Parameter import Parameter
import numpy
import logging

class NaiveBayes(AbstractPredictor):
    def __init__(self):
        self.featureSets = []
        self.condMatrices = [] 
        self.labelSet = 0
    
    def __findConditionalProb(self, X, y, featureIndex):
        examples = X
        feature = numpy.array([examples[:, featureIndex]]).T
        labels = y
        
        featureSet = numpy.unique(feature) 
        labelSet = numpy.unique(labels)
        
        condMatrix = numpy.zeros((labelSet.shape[0], featureSet.shape[0]))
        
        for i in range(labelSet.shape[0]):
            for j in range(featureSet.shape[0]): 
                probLabelAndFeature = sum(numpy.logical_and(labels == labelSet[i], feature == featureSet[j]))
                probFeature = sum(feature == featureSet[j])
                
                condMatrix[i, j] = float(probLabelAndFeature)/float(probFeature)
        
        return condMatrix, featureSet, labelSet
    
    #Here we store a list of arrays of conditional probability matrices 
    def learnModel(self, X, y):
        Parameter.checkClass(X, numpy.ndarray)
        Parameter.checkClass(y, numpy.ndarray)

        if len(self.featureSets) != 0 or len(self.condMatrices) != 0: 
            raise ValueError("Cannot learn on an already trained object")
        
        numExamples = X.shape[0]
        numFeatures = X.shape[1]
        
        for i in range(numFeatures): 
            condMatrix, featureSet, self.labelSet = self.__findConditionalProb(X, y, i)
            
            #Note that the columns of the condition matrix sum to 1 
            if not (condMatrix.sum(axis=0) == numpy.ones(featureSet.shape[0])).all(): 
                raise ValueError("Columns of condition matrix do not sum to 1")
            
            self.condMatrices.append(condMatrix)
            self.featureSets.append(featureSet)
            
        logging.info("Learnt Naive Bayes Model using " + str(numExamples) + " examples and " + str(numFeatures) + " features.")

    """
    This is probably a bit slow 
    We assume all feature values in the test set are also present in the training set 
    """
    def classify(self, X):
        Parameter.checkClass(X, numpy.ndarray)
        if len(self.featureSets) == 0 and len(self.condMatrices) == 0: 
            raise ValueError("Must train before classification.")
        
        numExamples = X.shape[0]
        numFeatures = X.shape[1]
        

        y = numpy.zeros((numExamples))
        pys = numpy.zeros((numExamples))
        
        for i in range(numExamples): 
            #The probabilities of all choices of y 
            currentPy = numpy.ones((self.labelSet.shape[0]))
            
            for j in range(numFeatures): 
                if X[i, j] not in self.featureSets[j]:
                    #If the feature was not in the training data assume uniform probability
                    pYgivenXj = numpy.ones((self.labelSet.shape[0]))/self.labelSet.shape[0]
                else:
                    fIndex = self.featureSets[j].tolist().index(X[i, j])
                    pYgivenXj = self.condMatrices[j][:, fIndex]
                
                currentPy  = currentPy * pYgivenXj
                
            pyIndex = numpy.argmax(currentPy)
            y[i] = self.labelSet[pyIndex]
            pys[i] = currentPy[pyIndex]
        
        logging.info("Classified " + str(numExamples) + " examples and " + str(numFeatures) + " features.")
        self.pys = pys
        
        return y 
    
    def getProbabilities(self): 
        return self.pys
    
    def getCondMatrix(self, i): 
        return self.condMatrices[i]
    
    def getFeatureSet(self, i): 
        return self.featureSets[i]
    
    def getLabelSet(self):
        return self.labelSet
    
    featureSets = []
    condMatrices = [] 
    labelSet = 0
    pys = 0 
    