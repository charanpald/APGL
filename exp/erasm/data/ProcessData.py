"""
Process the data from mendeley into a series of matrices. 
"""
from apgl.graph import DictGraph  
from apgl.util.PathDefaults import PathDefaults 
from exp.sandbox.igraph.GraphStatistics import GraphStatistics
import logging 
import sys 
import itertools 
import igraph 
import matplotlib.pyplot as plt 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
path = "/local/dhanjalc/dataDump-28-11-12/" 
path = PathDefaults.getDataDir() + "erasm/"

def contactsGraph(): 
    fileName = path + "connections-28-11-12"
    vertexIdDict = {} 
    vertexIdSet = set([])
    edgeSet = set([])
    edgeArray = []
    graph = igraph.Graph()
    i = 0 
    j = 0 
    
    with open(fileName) as f:
        f.readline()  
        
        for line in f:
            if i % 50000 == 0: 
                print(i)
            words = line.split()
            vId1 = int(words[0])
            vId2 = int(words[1])
            
            if vId1 not in vertexIdSet:    
                vertexIdDict[vId1] = j 
                vertexIdSet.add(vId1)
                j += 1 
            
            if vId2 not in vertexIdSet:    
                vertexIdDict[vId2] = j 
                vertexIdSet.add(vId2)
                j += 1 
            
            if (vertexIdDict[vId1], vertexIdDict[vId2]) not in edgeSet and (vertexIdDict[vId2], vertexIdDict[vId1]) not in edgeSet: 
                edgeArray.append([vertexIdDict[vId1], vertexIdDict[vId2]])
                edgeSet.add((vertexIdDict[vId1], vertexIdDict[vId2]))
                
            i += 1 

    print("Read " + str(i) + " lines with " + str(j) + " vertices")     
    
    graph.add_vertices(j)
    graph.add_edges(edgeArray)    
    print(igraph.summary(graph))

    graphStats = GraphStatistics()
    statsArray = graphStats.scalarStatistics(graph, slowStats=False)    
    print(graphStats.strScalarStatsArray(statsArray))
    
    xs, ys = zip(*[(left, count) for left, _, count in graph.degree_distribution().bins()])
    plt.figure(0)
    plt.bar(xs[0:30], ys[0:30])
    plt.xlabel("Degree")

    xs, ys = zip(*[(left, count) for left, _, count in graph.components().size_histogram().bins()])
    plt.figure(1)
    plt.bar(xs[0:30], ys[0:30])
    plt.xlabel("Component size")
    plt.show()
    
    
def readBipartiteGraph(fileName):
    groupSet = set([])
    groupsDict = {} 
    vertexIdDict = {} 
    vertexIdSet = set([])
    edgeSet = set([])
    edgeArray = []
    i = 0 
    
    with open(fileName) as f:
        f.readline()        
        
        for line in f:
            if i % 50000 == 0: 
                print(i)                    
            
            words = line.split()
            if words[0] not in groupSet: 
                groupsDict[words[0]] = [words[1]]
                groupSet.add(words[0])
            else: 
                groupsDict[words[0]].append(words[1])
            i += 1 

    print("Read " + str(i) + " lines")     

    #Now link people in the same group 
    k = 0  
    count = 0 
    
    for key, value in groupsDict.items(): 
        iterator = itertools.permutations(range(len(value)), 2)
        
        for i, j in iterator: 
            if count % 200000 == 0: 
                print(count)                 
            
            vId1 = value[i]
            vId2 = value[j]
        
            if vId1 not in vertexIdSet:    
                vertexIdDict[vId1] = k 
                vertexIdSet.add(vId1)
                k += 1 
            
            if vId2 not in vertexIdSet:    
                vertexIdDict[vId2] = k 
                vertexIdSet.add(vId2)
                k += 1 
                
            if (vertexIdDict[vId1], vertexIdDict[vId2]) not in edgeSet and (vertexIdDict[vId2], vertexIdDict[vId1]) not in edgeSet: 
                edgeArray.append([vertexIdDict[vId1], vertexIdDict[vId2]])
                edgeSet.add((vertexIdDict[vId1], vertexIdDict[vId2]))
            
            count += 1 
        
    graph = igraph.Graph()
    graph.add_vertices(len(vertexIdSet))
    graph.add_edges(edgeArray) 
    
    return graph
    
    
def groupsGraph(): 
    fileName = path + "groupMembers-29-11-12"
    graph = readBipartiteGraph(fileName) 
    print(igraph.summary(graph))

    graphStats = GraphStatistics()
    statsArray = graphStats.scalarStatistics(graph, slowStats=False)    
    print(graphStats.strScalarStatsArray(statsArray))
    
    xs, ys = zip(*[(left, count) for left, _, count in graph.degree_distribution().bins()])
    plt.figure(0)
    plt.bar(xs[0:30], ys[0:30])
    plt.xlabel("Degree")

    xs, ys = zip(*[(left, count) for left, _, count in graph.components().size_histogram().bins()])
    plt.figure(1)
    plt.bar(xs[0:30], ys[0:30])
    plt.xlabel("Component size")
    plt.show()    

def articlesGraph(): 
    fileName = path + "articleMendeleyAuthors-28-11-12"
    graph = readBipartiteGraph(fileName) 
    print(igraph.summary(graph))

    graphStats = GraphStatistics()
    statsArray = graphStats.scalarStatistics(graph, slowStats=False)    
    print(graphStats.strScalarStatsArray(statsArray))
    
    xs, ys = zip(*[(left, count) for left, _, count in graph.degree_distribution().bins()])
    plt.figure(0)
    plt.bar(xs[0:30], ys[0:30])
    plt.xlabel("Degree")

    xs, ys = zip(*[(left, count) for left, _, count in graph.components().size_histogram().bins()])
    plt.figure(1)
    plt.bar(xs[0:30], ys[0:30])
    plt.xlabel("Component size")
    plt.show()   
    
def articleGroupsGraph(): 
    fileName = path + "articleGroupMembership-28-11-12"
    graph = readBipartiteGraph(fileName) 
    print(igraph.summary(graph))

    graphStats = GraphStatistics()
    statsArray = graphStats.scalarStatistics(graph, slowStats=False)    
    print(graphStats.strScalarStatsArray(statsArray))
    
    xs, ys = zip(*[(left, count) for left, _, count in graph.degree_distribution().bins()])
    plt.figure(0)
    plt.bar(xs[0:30], ys[0:30])
    plt.xlabel("Degree")

    xs, ys = zip(*[(left, count) for left, _, count in graph.components().size_histogram().bins()])
    plt.figure(1)
    plt.bar(xs[0:30], ys[0:30])
    plt.xlabel("Component size")
    plt.show() 

def fullCoauthorGraph(): 
    fileName = path + "coauthorsGraph"
    graph = igraph.Graph()
    graph = graph.Read_Edgelist(fileName)
    graph = graph.as_undirected()
    print(igraph.summary(graph))

    graphStats = GraphStatistics()
    statsArray = graphStats.scalarStatistics(graph, slowStats=False)    
    print(graphStats.strScalarStatsArray(statsArray))

#contactsGraph()
#groupsGraph() 
#articlesGraph()
#articleGroupsGraph()
fullCoauthorGraph() 
