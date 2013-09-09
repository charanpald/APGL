
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
graphRanker = GraphRanker(k=100, numRuns=100, computeInfluence=computeInfluence, p=0.05)
methodNames = graphRanker.getNames()

numMethods = 7
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
        outputLists, trainExpertMatchesInds, testExpertMatchesInds = Util.loadPickle(outputFilename)
        graph, authorIndexer = Util.loadPickle(dataset.getCoauthorsFilename(field))
        
        numMethods = len(outputLists)
        precisions = numpy.zeros((len(ns), numMethods))
        
        f1Scores = numpy.zeros(numMethods)
        
        for i, n in enumerate(ns):     
            for j in range(len(outputLists)): 
                newOutputList = []
                for item in outputLists[j][0:n]: 
                    if item not in trainExpertMatchesInds: 
                        newOutputList.append(item)
                
                precisions[i, j] = Evaluator.precisionFromIndLists(testExpertMatchesInds, newOutputList) 
                averagePrecisions[s, i, j] = Evaluator.averagePrecisionFromLists(testExpertMatchesInds, newOutputList, n) 
        
        print(field)      
        print(authorIndexer.reverseTranslate(outputLists[-1][0:10]))
        print(authorIndexer.reverseTranslate(testExpertMatchesInds))
        print(precisions)
        print(averagePrecisions[s, :, :] )
    except IOError as e: 
        print(e)

meanAveragePrecisions = numpy.mean(averagePrecisions, 0)
meanAveragePrecisions = numpy.c_[numpy.array(ns), meanAveragePrecisions]
print("==== Summary ====")
print(Latex.listToRow(methodNames))
print(Latex.array2DToRows(meanAveragePrecisions))

