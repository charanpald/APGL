"""
Process the data from mendeley into a series of matrices. 
"""
from apgl.graph import SparseGraph, DictGraph, GraphStatistics  
import logging 
import sys 
import itertools 

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
    graphStats = GraphStatistics()
    statsArray = graphStats.scalarStatistics(graph, slowStats=False)
    
    print(graphStats.strScalarStatsArray(statsArray))
  
def groupsGraph(): 
    fileName = path + "groupMembers-29-11-12"
    groupsDict = {} 
    graph = DictGraph()
    i = 0 
    
    with open(fileName) as f:
        f.readline()        
        
        for line in f:
            words = line.split()
            if int(words[0]) not in groupsDict.keys(): 
                groupsDict[int(words[0])] = [int(words[1])]
            else: 
                groupsDict[int(words[0])].append(int(words[1]))
            i += 1 

    print("Read " + str(i) + " lines")     

    #Now link people in the same group 
    for key, value in groupsDict.items(): 
        iterator = itertools.permutations(range(len(value)), 2)
        
        for i, j in iterator:         
            graph[value[i], value[j]] = 1
        
    print(graph)
    
    graph = graph.toSparseGraph()
    print("Converted to SparseGraph, computing graph statistics")
    
    print(graph)
    graphStats = GraphStatistics()
    statsArray = graphStats.scalarStatistics(graph, slowStats=False)
    
    print(graphStats.strScalarStatsArray(statsArray))
  
#contactsGraph()
groupsGraph() 