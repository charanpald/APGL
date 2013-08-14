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
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        
        field = "Database"
        self.dataset = ArnetMinerDataset(field)
        self.dataset.dataFilename = self.dataset.dataDir + "DBLP-citation-test.txt"
        
    def testVectoriseDocuments(self): 
        #Check document is correct as well as authors 
        self.dataset.vectoriseDocuments()
        
    def testFindSimilarDocuments(self): 
        field = "Database"
        self.dataset = ArnetMinerDataset(field)
        self.dataset.dataFilename = self.dataset.dataDir + "DBLP-citation-test.txt"
        
        #Check document is correct as well as authors 
        self.dataset.vectoriseDocuments()
        self.dataset.findSimilarDocuments()
        
        experts = Util.loadPickle(self.dataset.relevantExpertsFilename)
        
        self.assertEquals(['Hector Garcia-Molina', 'Meichun Hsu'], experts[0:2])
        
    def testFindCoauthors(self): 
        
        #Check document is correct as well as authors 
        self.dataset.vectoriseDocuments()
        self.dataset.findSimilarDocuments()
        self.dataset.coauthorsGraph()
  

    def testCoauthorsGraphFromAuthors(self): 
        releventExperts = set(["Yuri Breitbart", "Hector Garcia-Molina"])
        
        graph, authorIndexer = self.dataset.coauthorsGraphFromAuthors(releventExperts)

        self.assertEquals(graph.get_edgelist(), [(0, 1), (0, 2), (0, 4), (1, 2), (1, 3)]) 
        
        self.assertEquals(graph.es["weight"], [1, 1, 2, 1, 1])
        self.assertEquals(graph.es["invWeight"], [1 ,1,0.5,1,1])
        
        self.assertEquals(len(authorIndexer.getIdDict()), 5)
       
       
    def testMatchExperts(self): 
        #TODO: 
        expertMatches, expertsSet = self.dataset.matchExperts() 
        
        self.assertEquals(expertMatches, ['Hector Garcia-Molina'])
        self.assertEquals(expertsSet, set(['Hector L. Garcia-Molina', 'Jimmy Conners']))
            
    def testExpertsFromDocSimilarities(self):
        self.dataset.authorList = [["Joe Bloggs", "Alfred Nobel"], ["Ian Hislop"], ["Alfred Nobel", "Ian Hislop"]]
        similarities = numpy.array([0.4, 0.5, 0.8]) 
        
        experts = self.dataset.expertsFromDocSimilarities(similarities)
        self.assertEquals(experts, ['Ian Hislop', 'Alfred Nobel', 'Joe Bloggs'])
       
if __name__ == '__main__':
    unittest.main()
