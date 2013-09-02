
import numpy 
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 
from exp.influence2.ArnetMinerDataset import ArnetMinerDataset
from apgl.util.Latex import Latex 
from apgl.util.Util import Util 
from apgl.util.Evaluator import Evaluator 

numpy.set_printoptions(suppress=True, precision=3, linewidth=100)
dataset = ArnetMinerDataset()

ns = numpy.arange(5, 55, 5)
averagePrecisionN = 20 
bestPrecisions = numpy.zeros((len(ns), len(dataset.fields)))
bestAveragePrecisions = numpy.zeros(len(dataset.fields))

for i, field in enumerate(dataset.fields): 
    outputFilename = dataset.getResultsDir(field) + "outputLists.npz"
    outputLists, expertMatchesInds = Util.loadPickle(outputFilename)
    
    numMethods = len(outputLists)
    precisions = numpy.zeros((len(ns), numMethods))
    averagePrecisions = numpy.zeros(numMethods)
    
    for i, n in enumerate(ns):     
        for j in range(len(outputLists)): 
            precisions[i, j] = Evaluator.precisionFromIndLists(expertMatchesInds, outputLists[j][0:n]) 
        
    for j in range(len(outputLists)):                 
        averagePrecisions[j] = Evaluator.averagePrecisionFromLists(expertMatchesInds, outputLists[j][0:averagePrecisionN], averagePrecisionN) 
    
    print(field)
    print(precisions)
    print(averagePrecisions)
    
    bestInd = numpy.argmax(averagePrecisions)
    plt.plot(ns, precisions[:, bestInd], label=field)
    bestPrecisions[:, i] = precisions[:, bestInd]
    bestAveragePrecisions[i] = averagePrecisions[bestInd]

bestPrecisions2 = numpy.c_[numpy.array(ns), bestPrecisions]
print(Latex.array2DToRows(bestPrecisions2))
print(Latex.array1DToRow(bestAveragePrecisions))

print(dataset.fields)

plt.legend()
plt.show()