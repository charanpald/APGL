"""
Some code to profile the greedy influence function 
"""

import numpy
import logging
import sys
import cProfile
import pstats
import os 
from apgl.influence.GreedyInfluence import GreedyInfluence
from apgl.util import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

numVertices = 200
P = numpy.random.rand(numVertices, numVertices)
k = 100 

outputDirectory = PathDefaults.getOutputDir()
directory = outputDirectory + "test/"
profileFileName = directory + "profile.cprof"

influence = GreedyInfluence()

cProfile.runctx('influence.maxInfluence(P, k)', globals(), locals(), profileFileName)
stats = pstats.Stats(profileFileName)
stats.strip_dirs().sort_stats("cumulative").print_stats(20)
