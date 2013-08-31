
import numpy 
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 
from exp.influence2.ArnetMinerDataset import ArnetMinerDataset
from apgl.util.Latex import Latex 

numpy.set_printoptions(suppress=True, precision=3, linewidth=100)
dataset = ArnetMinerDataset()

ns = numpy.arange(5, 55, 5)
precisionArray = numpy.zeros((len(ns), len(dataset.fields)))
averagePrecisionsArray = numpy.zeros(len(dataset.fields))

for i, field in enumerate(dataset.fields): 
    resultsFilename = dataset.getResultsDir(field) + "precisions.npz" 
    
    data = numpy.load(resultsFilename)
    precisions, averagePrecisions = data["arr_0"], data["arr_1"]
    bestInd = numpy.argmax(averagePrecisions)
    
    plt.plot(ns, precisions[:, bestInd], label=field)
    
    precisionArray[:, i] = precisions[:, bestInd]
    averagePrecisionsArray[i] = averagePrecisions[bestInd]

print(Latex.array2DToRows(precisionArray))
print(Latex.array1DToRow(averagePrecisionsArray))

print(dataset.fields)

plt.legend()
plt.show()