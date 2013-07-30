import numpy 
try:  
    ctypes.cdll.LoadLibrary("/usr/local/lib/libigraph.so")
except: 
    pass 
import igraph 
import logging 
import array 
import itertools
import difflib 
from apgl.util.PathDefaults import PathDefaults 
from exp.util.IdIndexer import IdIndexer 


class GraphReader(object): 
    """
    A class to take a set of publication files and create the corresponding graphs. 
    """
    def __init__(self, field): 
        self.field = field
        
        dirName = PathDefaults.getDataDir() + "reputation/" + self.field + "/"
        self.coauthorFilename = dirName + self.field.lower() + ".csv"
        self.trainExpertsFilename = dirName + self.field.lower() + "_seed_train" + ".csv"
        self.testExpertsFilename = dirName + self.field.lower() + "_seed_test" + ".csv"
        
        logging.debug("Publications filename: " + self.coauthorFilename)
        logging.debug("Training experts filename: " + self.trainExpertsFilename)
        logging.debug("Test experts filename: " + self.testExpertsFilename)
        
    def read(self): 
        coauthorFile = open(self.coauthorFilename)
        
        self.authorIndexer = IdIndexer("i")
        articleIndexer = IdIndexer("i")
        i = 0
        
        for line in coauthorFile: 
            vals = line.split(";")
            
            authorId = vals[0].strip().strip("=")
            articleId = vals[1].strip()
            
            self.authorIndexer.append(authorId)
            articleIndexer.append(articleId)
            i += 1 
            
        logging.debug("Total lines read: " + str(i))
        
        authorInds = self.authorIndexer.getArray()
        articleInds = articleIndexer.getArray()
        edges = numpy.c_[authorInds, articleInds]
        
        logging.debug("Number of unique authors: " + str(len(self.authorIndexer.getIdDict())))
        logging.debug("Number of unique articles: " + str(len(articleIndexer.getIdDict())))
        
        author1Inds = array.array('i')
        author2Inds = array.array('i')
        
        lastArticleInd = -1
        coauthorList = []

        #Go through and create coauthor graph 
        for i in range(authorInds.shape[0]): 
            authorInd = authorInds[i]    
            articleInd = articleInds[i] 
            
            if articleInd != lastArticleInd:         
                iterator = itertools.combinations(coauthorList, 2)
                for vId1, vId2 in iterator:   
                    author1Inds.append(vId1)
                    author2Inds.append(vId2)
                
                coauthorList = []
                coauthorList.append(authorInd)
            else: 
                coauthorList.append(authorInd)
                
            lastArticleInd = articleInd

        author1Inds = numpy.array(author1Inds, numpy.int)
        author2Inds = numpy.array(author2Inds, numpy.int)
        edges = numpy.c_[author1Inds, author2Inds]
        
        #Coauthor graph is undirected 
        graph = igraph.Graph()
        graph.add_vertices(numpy.max(authorInds)+1)
        graph.add_edges(edges)
        
        logging.debug(graph.summary())
        
        logging.debug("Number of components in graph: " + str(len(graph.components()))) 
        compSizes = [len(x) for x in graph.components()]
        logging.debug("Max component size: " + str(numpy.max(compSizes))) 
        
        return graph
        
    def readExperts(self): 
        #Now load list of experts
        expertsFile = open(self.testExpertsFilename)
        expertsList = []
        expertsIdList = []  
        i = 0
        
        for line in expertsFile: 
            vals = line.split() 
            key = vals[0][0].lower() + "/" + vals[0].strip(",") + ":" 
            expertName = vals[-1] + ", "    
            
            for j in range(1, len(vals)): 
                if j != len(vals)-2:
                    key += vals[j].strip(".,") + "_"
                else: 
                    key += vals[j].strip(".,")
                    
                expertName += vals[j] + " "
                
            key = key.strip()
            expertName = expertName.strip()
                
            possibleExperts = difflib.get_close_matches(key, self.authorIndexer.getIdDict().keys(), cutoff=0.8)        
                
            if len(possibleExperts) != 0:
                #logging.debug("Matched key : " + key + ", " + possibleExperts[0])
                expertsIdList.append(self.authorIndexer.getIdDict()[possibleExperts[0]])
                expertsList.append(expertName) 
                
            else: 
                logging.debug("Key not found : " + line.strip() + ": " + key)
                possibleExperts = difflib.get_close_matches(key, self.authorIndexer.getIdDict().keys(), cutoff=0.6) 
                logging.debug("Possible matches: " + str(possibleExperts))
                
            i += 1 
        expertsFile.close()
        logging.debug("Found " + str(len(expertsIdList)) + " of " + str(i) + " experts")
            
        return expertsList, expertsIdList 