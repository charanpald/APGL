
import numpy 
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 
from exp.influence2.ArnetMinerDataset import ArnetMinerDataset
from apgl.util.Latex import Latex 
from apgl.util.Util import Util 
from apgl.util.Evaluator import Evaluator 

ranLSI = True
numpy.set_printoptions(suppress=True, precision=3, linewidth=100)
dataset = ArnetMinerDataset(runLSI=ranLSI)

ns = numpy.arange(5, 55, 5)
averagePrecisionN = 20 
bestPrecisions = numpy.zeros((len(ns), len(dataset.fields)))
bestAveragePrecisions = numpy.zeros(len(dataset.fields))
bestF1Scores = numpy.zeros(len(dataset.fields))

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
        averagePrecisions = numpy.zeros(numMethods)
        f1Scores = numpy.zeros(numMethods)
        
        for i, n in enumerate(ns):     
            for j in range(len(outputLists)): 
                precisions[i, j] = Evaluator.precisionFromIndLists(expertMatchesInds, outputLists[j][0:n]) 
            
        for j in range(len(outputLists)):                 
            averagePrecisions[j] = Evaluator.averagePrecisionFromLists(expertMatchesInds, outputLists[j][0:averagePrecisionN], averagePrecisionN) 
            f1Scores[j] = Evaluator.f1FromIndLists(expertMatchesInds, outputLists[j][0:averagePrecisionN]) 
        
        print(field)
        print(precisions)
        print(averagePrecisions)
        print(f1Scores)
        
        bestInd = numpy.argmax(averagePrecisions)
        plt.plot(ns, precisions[:, bestInd], label=field)
        bestPrecisions[:, s] = precisions[:, bestInd]
        bestAveragePrecisions[s] = averagePrecisions[bestInd]
        bestF1Scores[s] = f1Scores[bestInd]
    except IOError as e: 
        print(e)

bestPrecisions2 = numpy.c_[numpy.array(ns), bestPrecisions]
print(Latex.array2DToRows(bestPrecisions2))
print(Latex.array1DToRow(bestAveragePrecisions))
print(Latex.array1DToRow(bestF1Scores))

print(dataset.fields)

plt.legend()
plt.show()