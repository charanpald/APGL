'''
Created on 3 Aug 2009

@author: charanpal
'''

import numpy
import logging 
from apgl.data.AbstractPreprocessor import AbstractPreprocessor
from apgl.util.Parameter import Parameter


class Standardiser(AbstractPreprocessor):
    '''
    A class which has some simple, but useful preprocessing function for 2D arrays. Once a preprocessing function
    is applied, it can be used on other data from the same source. 
    '''
    def __init__(self):
        '''
        Constructor
        '''
        self.normVector = None 
        self.centreVector = None
        self.minVals = None 

    def scaleArray(self, X):
        """
        Take an array X and scale so that the minimum and maximum values are -1
        and 1 respectively. 
        """

        if self.minVals == None: 
            minVals = numpy.amin(X, 0)
            maxVals = numpy.amax(X, 0)

            self.range = maxVals-minVals
            X = X*2/self.range

            self.minVals = numpy.amin(X, 0)
            X = X - self.minVals - 1
        else:
            X = X*2/self.range
            X = X - self.minVals - 1

        return X 

    def normaliseArray(self, X):
        """
        Normalise a set of examples by setting the norm of the columns to be 1. Return normalised array and 
        normalisation vector. 
        """
        
        if self.normVector == None:
            self.normVector = numpy.sqrt(numpy.sum(X**2, 0))
            self.normVector = self.normVector + (self.normVector == 0)
            
        return X/self.normVector
    
    def centreArray(self, X):
        """
        Centre an array by setting the sum of the columns to zero. 
        """ 
        
        if self.centreVector == None:
            self.centreVector = numpy.sum(X, 0)/X.shape[0]
            
        return (X - self.centreVector)
        
    def standardiseArray(self, X):
        """
        Centre and then normalise an array to have norm 1.
        """
        Parameter.checkClass(X, numpy.ndarray)

        X = self.centreArray(X)
        X = self.normaliseArray(X)

        logging.debug("Standardised array of shape " + str(X.shape))
        return X

    def unstandardiseArray(self, X):
        """
        Basically reverse the standardisation process. 
        """

        X = X * self.normVector
        X = X + self.centreVector
        return X 

    def learn(self, X):
        """
        Computes the mean  and normalisation vectors. 
        """
        self.normVector = None
        self.centreVector = None 
        return self.standardiseArray(X)

    
    def process(self, X):
        """ Inherited from AbstractPreprocessor """ 
        return self.standardiseArray(X)
    
    def getNormVector(self):
        return self.normVector
    
    def getCentreVector(self):
        return self.centreVector
    
    normVector = None 
    centreVector = None 