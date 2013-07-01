import numpy 
import heapq 
import logging

class MaxInfluence(object): 
    def __init__(self):
        pass 
    
    @staticmethod 
    def greedyMethod(graph, k, numRuns=10): 
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
                infSums[vertexInd]  = MaxInfluence.simulateCascades(graph, influenceSet.union([vertexInd]), numRuns, 21)  
            
            bestVertexInd = numpy.argmax(infSums)
            currentInfluence = infSums[bestVertexInd] 
            influenceSet.add(bestVertexInd)
            influenceList.append(bestVertexInd)
            unvisited.remove(bestVertexInd)
    
        return influenceList
                
    @staticmethod 
    def simulateCascade(graph, activeVertexInds): 
        allActiveVertexInds = activeVertexInds.copy() 
        currentActiveInds = activeVertexInds.copy()
        
        while len(currentActiveInds) != 0: 
            newActiveVertices = set([])            
            
            for vertexInd in currentActiveInds: 
                for vertexInd2 in graph.neighbors(vertexInd, mode="out"): 
                    if numpy.random.rand() <= graph.es["p"][graph.get_eid(vertexInd, vertexInd2)] and vertexInd2 not in allActiveVertexInds: 
                        newActiveVertices.add(vertexInd2)
            
            allActiveVertexInds = allActiveVertexInds.union(newActiveVertices)
            currentActiveInds = newActiveVertices
            
        return allActiveVertexInds 
            
    @staticmethod 
    def simulateCascades(graph, activeVertexInds, numRuns, seed=21): 
        numpy.random.seed(seed)        
        
        currentInfluence = 0 
        for j in range(numRuns): 
            currentInfluence += len(MaxInfluence.simulateCascade(graph, activeVertexInds))
        currentInfluence /= float(numRuns) 
        
        return currentInfluence 

    @staticmethod 
    def celf(graph, k, numRuns=10): 
        """
        Maximising the influence using the CELF algorithm of Leskovec et al. 
        """
        influenceSet = set([])
        influenceList = []                
        negMarginalIncreases = []
        
        #For the initial values we compute marginal increases with respect to the empty set 
        for vertexInd in range(graph.vcount()): 
            currentInfluence = MaxInfluence.simulateCascades(graph, influenceSet.union([vertexInd]), numRuns, 21)         
            #Note that we store the negation of the influence since heappop chooses the smallest value 
            heapq.heappush(negMarginalIncreases, (-currentInfluence, vertexInd))
                     
        negLastInfluence, bestVertexInd = heapq.heappop(negMarginalIncreases)
        influenceSet.add(bestVertexInd)
        influenceList.append(bestVertexInd)
                
        for i in range(1, k):
            logging.debug(i)
            valid = numpy.zeros(graph.vcount(), numpy.bool) 
            negMarginalInfluence, currentBestVertexInd = heapq.heappop(negMarginalIncreases)    
            
            while not valid[currentBestVertexInd]: 
                marginalInfluence = MaxInfluence.simulateCascades(graph, influenceSet.union([currentBestVertexInd]), numRuns, 21) 
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
        