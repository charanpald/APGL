
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
ks = numpy.array(2**numpy.arange(3, 8.5, 0.5), numpy.int)
logging.debug(ks)

plotStyles = ['k-', 'k--', 'k-.', 'r--', 'r-', 'g-', 'b-', 'b--', 'b-.', 'g--', 'g--', 'g-.', 'r-', 'r--', 'r-.']
i = 0

try: 
    fileName = outputDir + "ResultsSoftImpute.npz"
    data = numpy.load(fileName)
    measures = data["arr_0"]
    metadata = data["arr_1"]
    
    print(metadata[0,1])
    
    plt.figure(0)
    plt.plot(numpy.arange(measures.shape[0]), measures[:, 1], plotStyles[i],)
    plt.ylabel("RMSE Test")
    plt.legend(loc="lower left") 
    
    plt.figure(1)
    plt.plot(numpy.arange(measures.shape[0]), measures[:, 2], plotStyles[i])
    plt.ylabel("MAE Test")
    plt.legend(loc="lower left") 
    
    plt.figure(2)
    plt.plot(numpy.arange(measures.shape[0]), measures[:, 0], plotStyles[i])
    plt.legend() 
    plt.ylabel("RMSE Train")
    
    plt.figure(3)
    plt.plot(numpy.arange(metadata.shape[0]), metadata[:, 0], plotStyles[i])
    plt.legend() 
    plt.ylabel("Rank")
except: 
    logging.debug("Missing results : " + str(fileName))
   

plt.show()
