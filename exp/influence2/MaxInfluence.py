import numpy 
import heapq 
import logging

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
    def simulateCascade(graph, activeVertexInds, p=None): 
        allActiveVertexInds = activeVertexInds.copy() 
        currentActiveInds = activeVertexInds.copy()
        
        while len(currentActiveInds) != 0: 
            newActiveVertices = set([])            
            
            for vertexInd in currentActiveInds: 
                vertexSet =  set(graph.neighbors(vertexInd, mode="out")).difference(allActiveVertexInds)   
                
                if p==None: 
                    a = graph.get_eids([(vertexInd, vertexInd2) for vertexInd2 in vertexSet])
                    edgeVals = numpy.array(graph.es["p"])
                    edgePropagage = numpy.random.rand(len(vertexSet)) <=  edgeVals[a]
                else: 
                    edgePropagage = numpy.random.rand(len(vertexSet)) <= p 
                
                for i, vertexInd2 in enumerate(vertexSet): 
                    #if randNums[i] <= p[graph.get_eid(vertexInd, vertexInd2)]: 
                    if edgePropagage[i]: 
                        newActiveVertices.add(vertexInd2)
            
            allActiveVertexInds = allActiveVertexInds.union(newActiveVertices)
            currentActiveInds = newActiveVertices
            
        return allActiveVertexInds 
            
    @staticmethod 
    def simulateCascades(graph, activeVertexInds, numRuns, p=None, seed=21): 
        numpy.random.seed(seed)        
        
        currentInfluence = 0 
        for j in range(numRuns): 
            currentInfluence += len(MaxInfluence.simulateCascade(graph, activeVertexInds, p))
        currentInfluence /= float(numRuns) 
        
        return currentInfluence 

    @staticmethod 
    def celf(graph, k, numRuns=10, p=None): 
        """
        Maximising the influence using the CELF algorithm of Leskovec et al. 
        """
        influenceSet = set([])
        influenceList = []                
        negMarginalIncreases = []
        
        #For the initial values we compute marginal increases with respect to the empty set 
        for vertexInd in range(graph.vcount()): 
            currentInfluence = MaxInfluence.simulateCascades(graph, influenceSet.union([vertexInd]), numRuns, p, 21)         
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
                marginalInfluence = MaxInfluence.simulateCascades(graph, influenceSet.union([currentBestVertexInd]), numRuns, p, 21) 
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
        