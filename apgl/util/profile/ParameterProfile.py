import numpy
import logging
import sys
import cProfile
import pstats
import os

from apgl.graph import *
from apgl.util import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
numpy.random.seed(22)

class ParameterProfile():
    def __init__(self):
        outputDirectory = PathDefaults.getOutputDir()
        directory = outputDirectory + "test/"
        self.profileFileName = directory + "profile.cprof"

    def profileCheckInt(self):

        def runCheckInt():
            min = 0
            max = 10
            val = 5

            for i in range(300000):
                Parameter.checkInt(val, min, max)

        print("Starting to profile ... ")
        cProfile.runctx('runCheckInt()', globals(), locals(), self.profileFileName)
        stats = pstats.Stats(self.profileFileName)
        stats.strip_dirs().sort_stats("cumulative").print_stats(40)


profiler = ParameterProfile()
profiler.profileCheckInt()