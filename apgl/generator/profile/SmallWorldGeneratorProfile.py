from apgl.graph.SparseGraph import SparseGraph
from apgl.graph.VertexList import VertexList
from apgl.generator.SmallWorldGenerator import SmallWorldGenerator
from apgl.io.PajekWriter import PajekWriter
import unittest
import cProfile
import pstats


profileFileName = "profile.cprof"

p = 0.5
k = 15

numVertices = 200
numFeatures = 5

vList = VertexList(numVertices, numFeatures)
sGraph = SparseGraph(vList)
swg = SmallWorldGenerator(p, k)

cProfile.runctx('swg.generate(sGraph)', globals(), locals(), profileFileName)
stats = pstats.Stats(profileFileName)
stats.strip_dirs().sort_stats("cumulative").print_stats(20)