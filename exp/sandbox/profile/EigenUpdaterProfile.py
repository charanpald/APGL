import numpy
import logging
import sys
from apgl.graph import *
from apgl.generator import *
from apgl.util.ProfileUtils import ProfileUtils
from apgl.sandbox.EigenUpdater import *

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class EigenUpdaterProfile(object):
    def __init__(self):
        pass 


    def profileEigenConcat(self):
        k = 10
        n = 1000
        m = 100
        X = numpy.random.rand(n, n)

        XX = X.dot(X.T)
        self.AA = XX[0:m, 0:m]
        self.AB = XX[0:m, m:]
        self.BB = XX[m:, m:]

        self.omega, self.Q = numpy.linalg.eig(self.AA)

        ProfileUtils.profile('EigenUpdater.eigenConcat(self.omega, self.Q, self.AB, self.BB, k)', globals(), locals())

    def profileEigenAdd2(self):
        k = 10
        n = 1000
        m = 200
        X = numpy.random.rand(n, n)
        Y = numpy.random.rand(n, m)

        XX = X.dot(X.T)

        self.omega, self.Q = numpy.linalg.eig(XX)

        def runEigenAdd2():
            for i in range(10):
                EigenUpdater.eigenAdd2(self.omega, self.Q, Y, Y, k)

        ProfileUtils.profile('runEigenAdd2()', globals(), locals())

    def profileEigenRemove(self):
        k = 50
        n = 1000
        X = numpy.random.rand(n, n)
        m = 900

        XX = X.dot(X.T)
        self.omega, self.Q = numpy.linalg.eig(XX)

        def runEigenRemove():
            for i in range(10):
                EigenUpdater.eigenRemove(self.omega, self.Q, m, k)

        ProfileUtils.profile('runEigenRemove()', globals(), locals())


profiler = EigenUpdaterProfile()
#profiler.profileEigenConcat() #19.7
#profiler.profileEigenAdd2() #8.1
#The most costly are dot calls, but can't save much time
#Can look at sparse case 
profiler.profileEigenRemove() 