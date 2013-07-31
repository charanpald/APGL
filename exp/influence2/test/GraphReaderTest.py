import numpy 
import unittest
import logging
import sys 

from exp.influence2.GraphReader import GraphReader 

class  GraphReaderTest(unittest.TestCase):
    def setUp(self): 
        numpy.random.seed(22) 
        numpy.set_printoptions(suppress=True, precision=3)
        
        
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        
    def testRead(self): 
        field = "Test"
        reader = GraphReader(field) 
        graph = reader.read()
        
        self.assertEquals(len(reader.authorIndexer.getIdDict()), 4)
        self.assertEquals(len(reader.articleIndexer.getIdDict()), 3)
        
        self.assertEquals(graph.vcount(), 4)
        self.assertEquals(graph.ecount(), 4)
        
        edges = [i.tuple for i in graph.es()] 
        self.assertEquals(edges, [(0, 1), (0, 2), (1, 2), (1, 3)])
        
    def testReadExperts(self): 
        field = "Test"
        reader = GraphReader(field) 
        graph = reader.read()
        
        expertsList, expertsIdList = reader.readExperts()
        
        self.assertEquals(expertsIdList, [3, 1, 2])
        self.assertEquals(len(expertsList), 3)
        
if __name__ == '__main__':
    unittest.main()