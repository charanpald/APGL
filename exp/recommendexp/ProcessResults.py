
import numpy 
import matplotlib.pyplot as plt 
from apgl.util.PathDefaults import PathDefaults 


#For now just print some results for a particular dataset 
#dataset = "MovieLensDataset"
dataset = "NetflixDataset"
outputDir = PathDefaults.getOutputDir() + "recommend/" + dataset + "/"
ks = numpy.array([5, 10, 15, 20, 50, 100, 200, 300])

for k in ks: 
    data = numpy.load(outputDir + "ResultsSoftImpute_k=" + str(k) + ".npz")
    measures = data["arr_0"]
    print(measures)
    #plt.plot(numpy.arange(measures.shape[0]), measures[:, 1], label="k="+str(k))
    
#plt.show()
#plt.legend() 