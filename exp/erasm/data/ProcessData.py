"""
Process the data from mendeley into a series of matrices. 
"""
from apgl.graph import SparseGraph, DictGraph, GraphStatistics  
import logging 
import sys 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
path = "/local/dhanjalc/dataDump-28-11-12/" 

def contactsGraph(): 
    fileName = path + "connections-28-11-12"
    graph = DictGraph()
    i = 0 
    
    with open(fileName) as f:
        f.readline()        
        
        for line in f:
            words = line.split()
            graph[int(words[0]), int(words[1])] = 1
            i += 1 

    print("Read " + str(i) + " lines")         
    print(graph)
          
    graph = graph.toSparseGraph()
    print("Converted to SparseGraph, computing graph statistics")
    
    print(graph)
    print(type(graph.W))
    graphStats = GraphStatistics()
    statsArray = graphStats.scalarStatistics(graph, slowStats=False)
    
    
    print(graphStats.strScalarStatsArray(statsArray))
    
contactsGraph()