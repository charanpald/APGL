'''
Created on 1 Dec 2009

@author: charanpal
'''
#from apgl.features.AbstractExtractor import AbstractExtractor
from apgl.util.Util import Util 
import numpy
import math 

class MissingCompleter():
    """
    A kernel method to complete some missing data. 
    """
    def __init__(self):
        pass

    def learnModel(self, KHat, K, k):
        """
        A function to learn missing data based on Khat (partial kernel) and K
        (full kernel). 
        """

        numExamples = KHat.shape[0]
        Z = numpy.zeros((numExamples, numExamples))

        Ki = K.copy()
        alpha = numpy.zeros((numExamples, k))
        beta = numpy.zeros((numExamples, k))
        lmdba = numpy.zeros(k)
        a = 0.1

        for i in range(k):
            A = numpy.dot(numpy.dot(KHat, Ki), KHat)
            KHatSq = numpy.dot(KHat, KHat)
            KHatSqInv = numpy.linalg.inv(KHatSq)

            [D, V] = numpy.linalg.eig(numpy.dot(KHatSqInv, A))

            lmdba[i] = numpy.sqrt(D[0])
            alpha[:, i] = V[0:numExamples, 0]
            beta[:, i] = numpy.dot(KHat, alpha[:, i])

            #alpha[:, i] = alpha[:, i]/math.sqrt(Util.mdot((alpha[:, i].T, KHatSq, alpha[:, i])))

            #Note: Check this 
            beta[:, i] = beta[:, i]/numpy.sqrt(Util.mdot(beta[:, i].T, KHat, beta[:, i]))

            #Deflate Ki
            #print(Ki)
            Ki = Ki - Util.mdot(Ki, beta[:, i], beta[:, i].T, Ki.T)/Util.mdot(beta[:, i].T, Ki, beta[:, i])

            #print(Ki)
            
        return alpha, beta 

    def project(self, examplesList):
        pass