from apgl.util.PathDefaults import PathDefaults 
from exp.modelselect.ModelSelectUtils import ModelSelectUtils 
import matplotlib.pyplot as plt 
import logging 
import numpy 

outputDir = PathDefaults.getOutputDir() + "modelPenalisation/regression/CART/"

datasets = ModelSelectUtils.getRegressionDatasets(True)
gammas = numpy.unique(numpy.array(numpy.round(2**numpy.arange(1, 7.25, 0.25)-1), dtype=numpy.int))


print(gammas)

for datasetName, numRealisations in datasets:
    try: 
        A = numpy.load(outputDir + datasetName + "Beta.npz")["arr_0"]
        
        inds = gammas>10
        tempGamma = numpy.sqrt(gammas[inds])
        tempA = A[inds, :]
            
        plt.figure(0)
        plt.plot(tempGamma, tempA[:, 0], label="50")
        plt.plot(tempGamma, tempA[:, 1], label="100")
        plt.plot(tempGamma, tempA[:, 2], label="200")
        plt.legend()
        plt.title(datasetName)    
        plt.xlabel("gamma")
        plt.ylabel("Beta")   
        
        plt.show()
    except: 
        print("Dataset not found " + datasetName)