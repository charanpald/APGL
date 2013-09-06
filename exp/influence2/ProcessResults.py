
import numpy 
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 
from exp.influence2.ArnetMinerDataset import ArnetMinerDataset
from exp.influence2.GraphRanker import GraphRanker
from apgl.util.Latex import Latex 
from apgl.util.Util import Util 
from apgl.util.Evaluator import Evaluator 

ranLSI = True
numpy.set_printoptions(suppress=True, precision=3, linewidth=100)
dataset = ArnetMinerDataset(runLSI=ranLSI)

ns = numpy.arange(5, 55, 5)
bestAveragePrecisions = numpy.zeros(len(dataset.fields))

computeInfluence = True
graphRanker = GraphRanker(k=100, numRuns=100, computeInfluence=computeInfluence, p=0.05, trainExpertsIdList=[])
methodNames = graphRanker.getNames()

numMethods = 8
averagePrecisions = numpy.zeros((len(dataset.fields), len(ns), numMethods))

coverages = numpy.load(dataset.coverageFilename)
print("==== Coverages ====")
print(coverages)

for s, field in enumerate(dataset.fields): 
    if ranLSI: 
        outputFilename = dataset.getOutputFieldDir(field) + "outputListsLSI.npz"
    else: 
        outputFilename = dataset.getOutputFieldDir(field) + "outputListsLDA.npz"
        
    try: 
        outputLists, expertMatchesInds = Util.loadPickle(outputFilename)
        
        
        numMethods = len(outputLists)
        precisions = numpy.zeros((len(ns), numMethods))
        
        f1Scores = numpy.zeros(numMethods)
        
        for i, n in enumerate(ns):     
            for j in range(len(outputLists)): 
                precisions[i, j] = Evaluator.precisionFromIndLists(expertMatchesInds, outputLists[j][0:n]) 
                averagePrecisions[s, i, j] = Evaluator.averagePrecisionFromLists(expertMatchesInds, outputLists[j][0:n], n) 

        print(field)
        print(precisions)
        print(averagePrecisions[s, :, :] )
    except IOError as e: 
        print(e)

meanAveragePrecisions = numpy.mean(averagePrecisions, 0)
meanAveragePrecisions = numpy.c_[numpy.array(ns), meanAveragePrecisions]
print("==== Summary ====")
print(methodNames)
print(Latex.array2DToRows(meanAveragePrecisions))

