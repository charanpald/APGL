
import logging 
import sys 
import numpy 
import matplotlib.pyplot as plt 
from apgl.util.PathDefaults import PathDefaults 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

#For now just print some results for a particular dataset 
#dataset = "MovieLensDataset"
#dataset = "NetflixDataset"
dataset = "SyntheticDataset1"
outputDir = PathDefaults.getOutputDir() + "recommend/" + dataset + "/"

plotStyles = ['k-', 'k--', 'k-.', 'r--', 'r-', 'g-', 'b-', 'b--', 'b-.', 'g--', 'g--', 'g-.', 'r-', 'r--', 'r-.']
methods = ["propack", "arpack", "rsvd", "svdUpdate", "rsvdUpdate"]

for i, method in enumerate(methods): 

    try: 
        fileName = outputDir + "ResultsSoftImpute_alg=" + method + ".npz"
        data = numpy.load(fileName)
        measures = data["arr_0"]
        metadata = data["arr_1"]
        
        print("lambda = " + str(metadata[0,1]) +  " rank = " + str(metadata[0, 0])) 
        
        plt.figure(0)
        plt.plot(numpy.arange(measures.shape[0]), measures[:, 0], plotStyles[i], label=method)
        plt.xlabel("Graph no.")        
        plt.ylabel("RMSE Test")
        plt.legend(loc="lower left") 
        
        plt.figure(1)
        plt.plot(numpy.arange(measures.shape[0]), measures[:, 1], plotStyles[i], label=method)
        plt.xlabel("Graph no.")        
        plt.ylabel("MAE Test")
        plt.legend(loc="lower left") 
        
        if measures.shape[1] == 3:
            plt.figure(2)
            plt.plot(numpy.arange(measures.shape[0]), measures[:, 2], plotStyles[i], label=method)
            plt.legend() 
            plt.xlabel("Graph no.")
            plt.ylabel("RMSE Train")
        
        plt.figure(3)
        plt.plot(numpy.arange(metadata.shape[0]), metadata[:, 0], plotStyles[i], label=method)
        plt.legend() 
        plt.xlabel("Graph no.")
        plt.ylabel("Rank")
        
        plt.figure(4)
        plt.plot(numpy.arange(metadata.shape[0]), numpy.cumsum(metadata[:, 2]), plotStyles[i], label=method)
        plt.legend() 
        plt.xlabel("Graph no.")
        plt.ylabel("Time (s)")
    except: 
        logging.debug("Missing results : " + str(fileName))
       

plt.show()
