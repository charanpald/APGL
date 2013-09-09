import numpy 
import unittest
import logging
import sys 
from exp.influence2.ArnetMinerDataset import ArnetMinerDataset 
from apgl.util.Util import Util 

class  ArnetMinerDatasetTest(unittest.TestCase):
    def setUp(self): 
        numpy.random.seed(22) 
        numpy.set_printoptions(suppress=True, precision=3)
        #logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
        
        self.field = "Database"
        self.dataset = ArnetMinerDataset(additionalFields=[self.field])
        self.dataset.dataFilename = self.dataset.dataDir + "DBLP-citation-test.txt"
        self.dataset.overwrite = True
        self.dataset.overwriteModel = True
        self.dataset.overwriteVectoriser = True        
        
    def testVectoriseDocuments(self): 
        #Check document is correct as well as authors 
        self.dataset.vectoriseDocuments()
        
    def testFindSimilarDocuments(self): 
        field = "Object"
        self.dataset = ArnetMinerDataset()
        self.dataset.dataFilename = self.dataset.dataDir + "DBLP-citation-test.txt"
        
        #Check document is correct as well as authors 
        self.dataset.vectoriseDocuments()
        relevantExperts = self.dataset.findSimilarDocumentsLSI(field)
        
        self.assertEquals(['Jos\xc3\xa9 A. Blakeley'], relevantExperts)
        
        #Let's test order of ranking on larger dataset
        print("Running on 10000 dataset")
        dataset = ArnetMinerDataset()
        dataset.minDf = 10**-6
        dataset.dataFilename = dataset.dataDir + "DBLP-citation-10000.txt"
        dataset.vectoriseDocuments()
        relevantExperts = dataset.findSimilarDocumentsLSI("Neural Networks")
        
        self.assertEquals(['Christopher M. Bishop', 'Michael I. Jordan', 'Fred L. Kitchens', 'Ai Cheo', 'Cesare Alippi', 'Giovanni Vanini', 'C. C. Taylor', 'David J. Spiegelhalter', 'Donald Michie'], relevantExperts)
        
    def testFindCoauthors(self): 
        
        #Check document is correct as well as authors 
        self.dataset.vectoriseDocuments()
        relevantExperts = self.dataset.findSimilarDocumentsLSI(self.field)
        self.dataset.coauthorsGraph(self.field, relevantExperts)
  

    def testCoauthorsGraphFromAuthors(self): 
        releventExperts = set(["Yuri Breitbart", "Hector Garcia-Molina"])
        
        graph, authorIndexer = self.dataset.coauthorsGraphFromAuthors(releventExperts)

        self.assertEquals(graph.get_edgelist(), [(0, 1), (0, 2), (0, 4), (1, 2), (1, 3)]) 
        
        self.assertEquals(graph.es["weight"], [1, 1, 1, 1, 1])
        self.assertEquals(graph.es["invWeight"], [1 ,1,1,1,1])
        
        self.assertEquals(len(authorIndexer.getIdDict()), 5)
       
       
    def testMatchExperts(self): 
        #TODO: 
        self.dataset.vectoriseDocuments()
        relevantExperts = self.dataset.findSimilarDocumentsLSI("DBMS")
        expertsSet = self.dataset.expertsDict[self.field]

        expertMatches = self.dataset.matchExperts(relevantExperts, expertsSet)
                     
        self.assertEquals(expertMatches, ['Nathan Goodman'])
        self.assertEquals(expertsSet, set(['Hector Garcia-Molina', 'Yuri Breitbart', 'Nathan Goodman']))
            
    def testExpertsFromDocSimilarities(self):
        self.dataset.authorList = [["Joe Bloggs", "Alfred Nobel"], ["Ian Hislop"], ["Alfred Nobel", "Ian Hislop"]]
        similarities = numpy.array([0.4, 0.5, 0.8]) 
        
        experts = self.dataset.expertsFromDocSimilarities(similarities)
        self.assertEquals(experts, ['Ian Hislop', 'Alfred Nobel', 'Joe Bloggs'])
        
    def testFindSimilarDocumentsLDA(self): 
        self.dataset = ArnetMinerDataset()
        self.dataset.dataFilename = self.dataset.dataDir + "DBLP-citation-1000.txt"
        self.dataset.overwrite = True
        self.dataset.overwriteModel = True
        self.dataset.overwriteVectoriser = True
        self.dataset.k = 20
        
        #Check document is correct as well as authors 
        self.dataset.findSimilarDocumentsLDA(self.field)

        #Let's test order of ranking on larger dataset
        print("Running on 10000 dataset using LDA")
        dataset = ArnetMinerDataset()
        dataset.minDf = 10**-5
        dataset.dataFilename = dataset.dataDir + "DBLP-citation-10000.txt"
        dataset.vectoriseDocuments()
        relevantExperts = dataset.findSimilarDocumentsLDA("Neural Networks")
        
        #self.assertEquals(['Christopher M. Bishop', 'Michael I. Jordan', 'Fred L. Kitchens', 'Ai Cheo', 'Cesare Alippi', 'Giovanni Vanini', 'C. C. Taylor', 'David J. Spiegelhalter', 'Donald Michie'], relevantExperts)

    @unittest.skip("")
    def testModelSelectionLSI(self): 
        self.dataset.dataFilename = self.dataset.dataDir + "DBLP-citation-1000.txt"
        self.dataset.overwrite = True
        self.dataset.overwriteModel = True
        self.dataset.overwriteVectoriser = True        
        
        self.dataset.vectoriseDocuments() 
        
        self.dataset.modelSelectionLSI()
  
if __name__ == '__main__':
    unittest.main()
