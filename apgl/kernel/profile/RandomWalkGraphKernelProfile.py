from apgl.graph import *
from apgl.kernel import *

import unittest
import numpy


numVertices = 30
numFeatures = 1
vList = VertexList(numVertices, numFeatures)

p = 0.2

g1 = DenseGraph(vList)
erg1 = ErdosRenyiGenerator(g1)
g1 = erg1.generateGraph(p)


g2 = DenseGraph(vList)
erg2 = ErdosRenyiGenerator(g2)
g2 = erg2.generateGraph(p)

linearKernel = LinearKernel()
tau = 1.0
lmbda = 0.01
pgk = RandWalkGraphKernel(lmbda)
#pgk = PermutationGraphKernel(tau, linearKernel)

directory = "output/test/"
profileFileName = directory + "profile.cprof"

cProfile.runctx('pgk.evaluate(g1, g2)', globals(), locals(), profileFileName)
stats = pstats.Stats(profileFileName)
stats.strip_dirs().sort_stats("cumulative").print_stats(20)