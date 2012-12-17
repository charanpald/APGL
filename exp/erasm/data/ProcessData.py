"""
Process the data from mendeley into a series of matrices. 
"""
from apgl.graph import SparseGraph, DictGraph 


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
            
    print(graph)
    print("Read " + str(i) + " lines")   
    
    components = graph.findConnectedComponents() 
    
contactsGraph()