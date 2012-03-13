import logging
import sys
import numpy
import random 
from apgl.egograph.SvmInfoExperiment import SvmInfoExperiment

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
numpy.random.seed(21)
random.seed(21)

SvmInfoExperiment.saveSvmParams(SvmInfoExperiment.getSvmParamsFileName())
