import numpy 
import heapq 
import logging
import itertools 
import multiprocessing 
try:  
    ctypes.cdll.LoadLibrary("/usr/local/lib/libigraph.so")
except: 
    pass 
import igraph 
from apgl.util.Util import Util 

def simulateCascadeSize(args): 
    graph, activeVertexInds, p, numRuns = args
    
    currentInfluence = 0 
    for j in range(numRuns): 
        currentInfluence += len(MaxInfluence.simulateCascade(graph, activeVertexInds, p))    

    return currentInfluence 
    
class MaxInfluence(object): 
    def __init__(self):
        pass 
    
    @staticmethod 
    def greedyMethod(graph, k, numRuns=10, p=None): 
        """
        Use a simple greedy algorithm to maximise influence. 
        """
        influenceSet = set([])
        influenceList = []        
        unvisited = set(range(graph.vcount()))  
        currentInfluence = 0 
                
        for i in range(k):
            logging.debug(i)
            infSums = numpy.zeros(graph.vcount()) 
            for vertexInd in unvisited: 
                infSums[vertexInd]  = MaxInfluence.simulateCascades(graph, influenceSet.union([vertexInd]), numRuns, p, 21)  
            
            bestVertexInd = numpy.argmax(infSums)
            currentInfluence = infSums[bestVertexInd] 
            influenceSet.add(bestVertexInd)
            influenceList.append(bestVertexInd)
            unvisited.remove(bestVertexInd)
    
        return influenceList

    
    @staticmethod 
    def simulateInitialCascade(graph, p=None):
        #First, figure out which edges are present in the percolation graph according 
        #to p 
        edges = numpy.arange(graph.ecount())[numpy.random.rand(graph.ecount()) <= p]       
        percolationGraph = graph.subgraph_edges(edges, delete_vertices=False)
        influences = numpy.zeros(percolationGraph.vcount())        
        components = percolationGraph.components()
        
        for component in components: 
            influences[component] = len(component)
        
        return influences 
           
    @staticmethod 
    def simulateCascade(graph, activeVertexInds, p=0.5): 
        allActiveVertexInds = activeVertexInds.copy() 
        currentActiveInds = activeVertexInds.copy()
            
        while len(currentActiveInds) != 0: 
            newActiveVertices = set([])            
            
            for vertexInd in currentActiveInds: 
                vertexSet =  set(graph.neighbors(vertexInd, mode="out")).difference(allActiveVertexInds)
                edgePropagage = numpy.random.rand(len(vertexSet)) <= p 
                
                for i, vertexInd2 in enumerate(vertexSet): 
                    #if randNums[i] <= p[graph.get_eid(vertexInd, vertexInd2)]: 
                    if edgePropagage[i]: 
                        newActiveVertices.add(vertexInd2)
            
            allActiveVertexInds = allActiveVertexInds.union(newActiveVertices)
            currentActiveInds = newActiveVertices
            
        return allActiveVertexInds 
   
    @staticmethod
    def simulateCascades(graph, activeVertexInds, numRuns, p=0.5, seed=None):
        if seed != None: 
            numpy.random.seed(seed)
        
        currentInfluence = 0
        for j in range(numRuns):
            currentInfluence += len(MaxInfluence.simulateCascade(graph, activeVertexInds, p))
        currentInfluence /= float(numRuns)
        
        return currentInfluence 
         
    @staticmethod 
    def simulateCascades2(graph, activeVertexInds, numRuns, p=0.5, seed=None):
        if seed != None: 
            numpy.random.seed(seed)        
        
        currentInfluence = 0 
        paramList = []
        for j in range(multiprocessing.cpu_count()): 
            paramList.append((graph.copy(), activeVertexInds.copy(), p, int(numRuns/multiprocessing.cpu_count())))
        
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        results = pool.imap(simulateCascadeSize, paramList, chunksize=1)  
        #results = itertools.imap(simulateCascadeSize, paramList)
        
        for item in results: 
            currentInfluence += item 
            
        pool.terminate()
            
        currentInfluence /= float(int(numRuns/multiprocessing.cpu_count())*multiprocessing.cpu_count()) 
        
        return currentInfluence 

    @staticmethod 
    def celf(graph, k, numRuns=100, p=0.5): 
        """
        Maximising the influence using the CELF algorithm of Leskovec et al. 
        """
        k = min(graph.vcount(), k)   
        
        influenceSet = set([])
        influenceList = []                
        negMarginalIncreases = []
        
        #For the initial values we compute marginal increases with respect to the empty set 
        influences = numpy.zeros(graph.vcount())
        
        for i in range(numRuns): 
            influences += MaxInfluence.simulateInitialCascade(graph, p=p)
        
        influences /= float(numRuns)  
        logging.debug("Simulated initial cascades")          
        
        for vertexInd in range(graph.vcount()):        
            #Note that we store the negation of the influence since heappop chooses the smallest value 
            heapq.heappush(negMarginalIncreases, (-influences[vertexInd], vertexInd))
        
        
        """
        for vertexInd in range(graph.vcount()):
            print(vertexInd)
            currentInfluence = MaxInfluence.simulateCascades(graph, influenceSet.union([vertexInd]), numRuns, p)
            #Note that we store the negation of the influence since heappop chooses the smallest value
            heapq.heappush(negMarginalIncreases, (-currentInfluence, vertexInd))
        """
           
        negLastInfluence, bestVertexInd = heapq.heappop(negMarginalIncreases)
        influenceSet.add(bestVertexInd)
        influenceList.append(bestVertexInd)
        logging.debug("Picking additional vertices")
                
        for i in range(1, k):
            Util.printIteration(i-1, 1, k-1)
            valid = numpy.zeros(graph.vcount(), numpy.bool) 
            negMarginalInfluence, currentBestVertexInd = heapq.heappop(negMarginalIncreases)    
            
            while not valid[currentBestVertexInd]: 
                marginalInfluence = MaxInfluence.simulateCascades(graph, influenceSet.union([currentBestVertexInd]), numRuns, p) 
                marginalInfluence += negLastInfluence 
                
                #Note that we store the negation of the influence since heappop chooses the smallest value 
                heapq.heappush(negMarginalIncreases, (-marginalInfluence, currentBestVertexInd)) 
                valid[currentBestVertexInd] = True
                
                negMarginalInfluence, currentBestVertexInd = heapq.heappop(negMarginalIncreases) 
                totalInfluence = -(negMarginalInfluence + negLastInfluence)
            
            negLastInfluence = -totalInfluence 
            
            influenceSet.add(currentBestVertexInd)
            influenceList.append(currentBestVertexInd)
        
        return influenceList
        