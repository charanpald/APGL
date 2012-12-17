"""
A set of benchmark tests in order to compare the speed of algorithms under different 
graph types. 
"""
import numpy 
import logging 
import time 
from apgl.graph import DenseGraph, SparseGraph 

numpy.set_printoptions(suppress=True, precision=3)

def generateEdges(): 
    edgeList = [] 
    
    density = 0.01 
    numVertices = numpy.array([500, 1000, 2000, 3000]) 
    
    for i in numVertices:  
        numEdges = (i**2) * density
        
        
        edges = numpy.zeros((numEdges, 2))
        edges[:, 0] = numpy.random.randint(0, i, numEdges)
        edges[:, 1] = numpy.random.randint(0, i, numEdges)
        
        edgeList.append((i, edges))
        
    return edgeList 
            
def benchmarkAddEdges(edgeList): 
    
    numGraphTypes = 2 
    timeArray = numpy.zeros((len(edgeList), numGraphTypes)) 
    i = 0 
    
    for numVertices, edges in edgeList: 
        print("Timing graphs of size " + str(numVertices) + " with " + str(edges.shape[0]) + " edges")
        
        print("Running DenseGraph benchmark")
        graph = DenseGraph(numVertices)        
        startTime = time.clock()
        graph.addEdges(edges)        
        timeArray[i, 0] =  time.clock() - startTime
        
        print("Running SparseGraph benchmark")
        graph = SparseGraph(numVertices)        
        startTime = time.clock()
        graph.addEdges(edges)        
        timeArray[i, 1] =  time.clock() - startTime
        
        i +=1 
            
    return timeArray 
    
edgeList = generateEdges() 
timeArray = benchmarkAddEdges(edgeList)

print(timeArray)