'''
Created on 18 Aug 2009

@author: charanpal

Take the data about Egos and Alters1 and generate transmissions. Save as a Matlab File. 
'''

from apgl.io import *
from apgl.util.Util import Util
from apgl.util.PathDefaults import PathDefaults
import logging
import sys
import random
import numpy
import csv

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
random.seed(21)
numpy.random.seed(21)

p = 0.7
missing = 0
eCsvReader = EgoCsvReader()
eCsvReader.setP(p)

dataDir = PathDefaults.getDataDir() + "infoDiffusion/"
egoFileName = dataDir + "EgoData.csv"
alterFileName = dataDir + "AlterData.csv"
examplesList, egoIndicesR, alterIndices, egoIndicesNR, alterIndicesNR = eCsvReader.readFiles(egoFileName, alterFileName, missing)


#Find all of the decays in information transmission 
egoTestFileName = dataDir + "EgoInfoTest.csv"
alterTestFileName = dataDir + "AltersInfoTest.csv"
decayGraph = eCsvReader.findInfoDecayGraph(egoTestFileName, alterTestFileName, egoIndicesR, alterIndices, egoIndicesNR, alterIndicesNR, egoFileName, alterFileName, missing)

logging.info("Size of decay graph: " + str(decayGraph.getNumVertices()))
logging.info("Number of edges: " + str(decayGraph.getNumEdges()))

#Now write the decays to a simplegraph file 
decayFileName = dataDir + "EgoAlterDecays.dat"
decayGraph.save(decayFileName)
logging.info("Wrote decays to file " + decayFileName)

#Now write out transmissions
sampleSize = 1000 
indices = Util.sampleWithoutReplacement(sampleSize, examplesList.getNumExamples())
examplesListSample = examplesList.getSubExamplesList(indices)

outputFileName1 = dataDir + "EgoAlterTransmissions"
outputFileName2 = dataDir + "EgoAlterTransmissions1000"

examplesList.writeToMatFile(outputFileName1)
examplesListSample.writeToMatFile(outputFileName2)


#Let's also write out the csv file for analysis
X = examplesList.getDataField(examplesList.getDefaultExamplesName())
y = examplesList.getDataField(examplesList.getLabelsName())

X = numpy.c_[X, y]

csvFileName = dataDir + "EgoAlterTransmissions.csv"

headings = eCsvReader.egoQuestionIds
headings.extend(eCsvReader.alterQuestionIds)

writer = csv.writer(open(csvFileName, "wb"))
writer.writerow(headings)
writer.writerows(X)

logging.info("Wrote transmissions to csv file " + csvFileName)

