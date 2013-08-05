try:  
    ctypes.cdll.LoadLibrary("/usr/local/lib/libigraph.so")
except: 
    pass 
import igraph 
import numpy 
from apgl.util.PathDefaults import PathDefaults 
import logging 

class GraphReader2(object): 
    """
    A class to read the similarity graph generated from the Arnetminer dataset 
    """
    def __init__(self, field): 
        self.field = field
        self.eps = 0.1
        
        dirName = PathDefaults.getDataDir() + "reputation/" + self.field + "/arnetminer/"
        self.coauthorFilename = dirName + "coauthors.csv"
        self.coauthorMatrixFilename = dirName + "coauthorSimilarity.npy"
        self.trainExpertsFilename = dirName + "experts_train_matches" + ".csv"
        self.testExpertsFilename = dirName + "experts_test_matches" + ".csv"
        
        logging.debug("Publications filename: " + self.coauthorFilename)
        logging.debug("Training experts filename: " + self.trainExpertsFilename)
        logging.debug("Test experts filename: " + self.testExpertsFilename)
        
    def read(self):
        
        K = numpy.load(self.coauthorMatrixFilename)
        K = K.tolist()
        graph = igraph.Graph.Weighted_Adjacency(K, mode="PLUS", loops=False)
        
        print(graph.summary())
        graph.simplify(combine_edges=sum)   
        graph.es["invWeight"] = 1.0/numpy.array(graph.es["weight"]) 
        
        return graph 
        
    def readExperts(self, train=False): 
        """
        Read the experts from a test file. Returns two lists: expertsList is the 
        list of their names, and expertsIdList is their integer ID. 
        """
        if not train:
            logging.debug("Reading test experts list")
            expertsFile = open(self.testExpertsFilename)
        else: 
            logging.debug("Reading training experts list")
            expertsFile = open(self.trainExpertsFilename)
            
        expertsList = expertsFile.readlines()
        expertsFile.close()
        
        coauthorsFile = open(self.coauthorFilename)
        coauthors = coauthorsFile.readlines() 
        coauthorsFile.close() 
        
        expertsIdList = []  
        
        for expert in expertsList: 
            if expert in coauthors: 
                expertsIdList.append(coauthors.index(expert))
        
        return expertsList, expertsIdList  