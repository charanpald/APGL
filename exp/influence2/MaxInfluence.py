import numpy 
import heapq 
import logging
import itertools 
import os
import multiprocessing 
try:  
    ctypes.cdll.LoadLibrary("/usr/local/lib/libigraph.so")
except: 
    pass 
import igraph 
from apgl.util.Util import Util 


def simulateAllCascades(args): 
    graph, activeVertexInds, p, j = args
    numpy.random.seed(j)
    influence = MaxInfluence.simulateAllCascades(graph, activeVertexInds, p=p)
        
    return influence 
    
class MaxInfluence(object): 
    def __init__(self):
        pass 
    
    @staticmethod 
    def greedyMethod(graph, k, numRuns=10, p=0.5, verbose=False): 
        """
        Use a simple greedy algorithm to maximise influence. 
        """
        
        influenceSet = set([])
        influenceList = []     
        influenceScores = []
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
            influenceScores.append(currentInfluence)
    
        if verbose: 
            return influenceList, influenceScores
        else: 
            return influenceList
          
    @staticmethod 
    def greedyMethod2(graph, k, numRuns=10, p=0.5, verbose=False): 
        """
        Use a simple greedy algorithm to maximise influence. In this case 
        we make use of simulateAllCascades 
        """
        
        influenceSet = set([])
        influenceList = []     
        influenceScores = []
        currentInfluence = 0 
        
        #Numpy messes up the CPU affinity, so here is how we fix it 
        os.system('taskset -p 0xffffffff %d' % os.getpid())
                
        for i in range(k):
            logging.debug(i)
            
            influences = MaxInfluence.parallelSimulateAllCascades2(graph, influenceSet, numRuns, p=p)

            influences[influenceList] = -1
            bestVertexInd = numpy.argmax(influences)
            currentInfluence = influences[bestVertexInd] 
            
            influenceSet.add(bestVertexInd)
            influenceList.append(bestVertexInd)
            influenceScores.append(currentInfluence)
    
        if verbose: 
            return influenceList, influenceScores
        else: 
            return influenceList

    @staticmethod 
    def parallelSimulateAllCascades(graph, activeVertexInds, numRuns, p): 
        influences = numpy.zeros(graph.vcount())
        
        for j in range(numRuns): 
            influences += MaxInfluence.simulateAllCascades(graph, activeVertexInds, p=p)
        influences /= float(numRuns)  
        
        return influences 
        
    @staticmethod 
    def parallelSimulateAllCascades2(graph, activeVertexInds, numRuns, p): 
        influences = numpy.zeros(graph.vcount())
        
        paramList = []
        for j in range(numRuns): 
            paramList.append((graph, activeVertexInds, p, j)) 
        
        chunksize = max(1, numRuns/multiprocessing.cpu_count())
        pool = multiprocessing.Pool()
        resultsList = pool.imap(simulateAllCascades, paramList, chunksize=chunksize)
        
        for result in resultsList: 
            influences += result 
            
        pool.close()
        
        influences /= float(numRuns)  
        
        return influences 
 
    @staticmethod 
    def simulateAllCascades(graph, activeVertexInds, p=0.5):
        """
        We work out the total influence after we add each vertex to the set 
        of active vertices. If the vertex is already in this set, no gain 
        will be made. 
        """        
        #Figure out which edges are present in the percolation graph according to p 
        edges = numpy.arange(graph.ecount())[numpy.random.rand(graph.ecount()) <= p]       
        percolationGraph = graph.subgraph_edges(edges, delete_vertices=False)
        influences = numpy.zeros(percolationGraph.vcount())     
        
        components = percolationGraph.components()
        
        if len(activeVertexInds) == 0: 
            for component in components: 
                influences[component] = len(component)
        else: 
            activeVertexInds = set(activeVertexInds)
            lastInfluence = 0             
            
            for component in components:
                if len(activeVertexInds.intersection(set(component))) == 0:  
                    influences[component] = len(component)
                else: 
                    lastInfluence += len(component)
                    
            influences += lastInfluence
        
        return influences

    @staticmethod 
    def simulateCascade(graph, activeVertexInds, p=0.5): 
        """
        Simulate a single cascade in the graph with the given active vertices. 
        """
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
        """
        Simulate runRuns cascades in this graph. 
        """
        if seed != None: 
            numpy.random.seed(seed)
        
        currentInfluence = 0
        for j in range(numRuns):
            currentInfluence += len(MaxInfluence.simulateCascade(graph, activeVertexInds, p))
        currentInfluence /= float(numRuns)
        
        return currentInfluence 
         
    @staticmethod 
    def celf(graph, k, numRuns=100, p=0.5, verbose=False): 
        """
        Maximising the influence using the CELF algorithm of Leskovec et al. 
        """
        k = min(graph.vcount(), k)   
        
        influenceSet = set([])
        influenceList = []    
        influenceScores = []            
        negMarginalIncreases = []
        
        
        #For the initial values we compute marginal increases with respect to the empty set 
        influences = numpy.zeros(graph.vcount())
        
        for i in range(numRuns): 
            influences += MaxInfluence.simulateAllCascades(graph, [], p=p)
        
        influences /= float(numRuns)  
        logging.debug("Simulated initial cascades")          
        
        for vertexInd in range(graph.vcount()):        
            #Note that we store the negation of the influence since heappop chooses the smallest value 
            heapq.heappush(negMarginalIncreases, (-influences[vertexInd], vertexInd))
        
        
        """
        for vertexInd in range(graph.vcount()):
            currentInfluence = MaxInfluence.simulateCascades(graph, influenceSet.union([vertexInd]), numRuns, p)
            #Note that we store the negation of the influence since heappop chooses the smallest value
            heapq.heappush(negMarginalIncreases, (-currentInfluence, vertexInd))
        """
           
        negLastInfluence, bestVertexInd = heapq.heappop(negMarginalIncreases)
        influenceSet.add(bestVertexInd)
        influenceList.append(bestVertexInd)
        influenceScores.append(-negLastInfluence)
        logging.debug("Picking additional vertices")
                
        for i in range(1, k):
            Util.printIteration(i-1, 1, k-1)
            valid = numpy.zeros(graph.vcount(), numpy.bool) 
            negMarginalInfluence, currentBestVertexInd = heapq.heappop(negMarginalIncreases)    
            
            j = 0             
            
            while not valid[currentBestVertexInd]: 
                marginalInfluence = MaxInfluence.simulateCascades(graph, influenceSet.union([currentBestVertexInd]), numRuns, p) 
                marginalInfluence += negLastInfluence 
                
                #Note that we store the negation of the influence since heappop chooses the smallest value 
                heapq.heappush(negMarginalIncreases, (-marginalInfluence, currentBestVertexInd)) 
                valid[currentBestVertexInd] = True
                
                negMarginalInfluence, currentBestVertexInd = heapq.heappop(negMarginalIncreases) 
                totalInfluence = -(negMarginalInfluence + negLastInfluence)
                j+=1 
                #print(j)
                
            logging.debug("Required " + str(j) + " evaluations to find influential vertex")
            
            negLastInfluence = -totalInfluence 
            
            influenceSet.add(currentBestVertexInd)
            influenceList.append(currentBestVertexInd)
            influenceScores.append(-negLastInfluence)
        
        if verbose: 
            return influenceList, influenceScores
        else: 
            return influenceList

        