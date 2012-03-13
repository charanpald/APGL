
import numpy
import logging
import sys
from apgl.graph import *
from apgl.util import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
numpy.random.seed(22)

class UtilProfile(object):
    def profileFitDiscretePowerLaw(self):
        #Test with a large vector x
        alpha = 2.5
        exponent = (1/(alpha-1))
        numPoints = 50000
        x = 10*numpy.random.rand(numPoints)**-exponent
        x = numpy.array(numpy.round(x), numpy.int)
        x = x[x<=500]
        x = x[x>=1]

        xmins = numpy.arange(1, 20)

        ProfileUtils.profile('Util.fitDiscretePowerLaw(x, xmins)', globals(), locals())

    def profileRandomChoice(self):
        n = 10000
        m = 1000

        maxInt = 20 
        v = numpy.random.randint(0, maxInt, n)

        def runRandomChoice():
            reps = 10000
            for i in range(reps):
                Util.randomChoice(v, m)

        ProfileUtils.profile('runRandomChoice()', globals(), locals())

    def profileRandom2Choice(self):
        n = 1000
        m = 1000

        V = numpy.random.rand(n, 2)

        def runRandom2Choice():
            reps = 100
            for i in range(reps):
                Util.randomChoice(V, m)

        ProfileUtils.profile('runRandom2Choice()', globals(), locals())

    def profileAltRandomChoice(self):
        n = 10000
        m = 1000
        maxInt = 20 
        v = numpy.random.randint(0, maxInt, n)
        
        def runRandomChoice():
            #can just do non-zero entries
            w = Util.expandIntArray(v)

            reps = 10000
            for i in range(reps):
                w[numpy.random.randint(0, w.shape[0])]

        ProfileUtils.profile('runRandomChoice()', globals(), locals())

profiler = UtilProfile()
#profiler.profileFitDiscretePowerLaw()
#profiler.profileRandomChoice()
#profiler.profileAltRandomChoice()
profiler.profileRandom2Choice()