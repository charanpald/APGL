'''
Created on 18 Aug 2009

@author: charanpal

Runs the graph simulation using the full set of Ego features. 
'''
import time
import logging
import sys
import os
import numpy
import random
from apgl.egograph.InfoExperiment import InfoExperiment

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
numpy.random.seed(21)
random.seed(21)

numVertices = SvmInfoExperiment.getNumVertices()

graphType = "SmallWorld"
ps = [0.01, 0.05, 0.1]
ks = [10, 15]
infoProbs = [0.1, 0.2, 0.5]

logging.info(time.strftime("Started at %a, %d %b %Y %H:%M:%S +0000", time.localtime()))
logging.info("Starting Small World Experiments")

simulator = SvmInfoExperiment.trainSVM()

for p in ps:
    for infoProb in infoProbs:
        for k in ks:
            if os.path.exists(SvmInfoExperiment.getOutputFileName(graphType, p, k, infoProb) + ".mat"):
                logging.info("Output file exists: " + SvmInfoExperiment.getOutputFileName(graphType, p, k, infoProb))
            else:
                SvmInfoExperiment.runExperiment(graphType, p, k, infoProb, simulator)

graphType = "ErdosRenyi"
ps = [10.0/numVertices, 20.0/numVertices, 30.0/numVertices, 40.0/numVertices]
logging.info("Starting Erdos-Renyi Experiments")

for p in ps:
    for infoProb in infoProbs:
        if os.path.exists(SvmInfoExperiment.getOutputFileName(graphType, p, k, infoProb) + ".mat"):
            logging.info("Output file exists: " + SvmInfoExperiment.getOutputFileName(graphType, p, k, infoProb))
        else:
            SvmInfoExperiment.runExperiment(graphType, p, k, infoProb, simulator)
            
logging.info(time.strftime("Ended at %a, %d %b %Y %H:%M:%S +0000", time.localtime()))