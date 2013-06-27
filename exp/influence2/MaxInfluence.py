import numpy 

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
                
        for i in range(k):
            print(i)
            infSums = numpy.zeros(graph.vcount()) 
            for vertexInd in unvisited: 
                for j in range(numRuns): 
                    infSums[vertexInd] += len(MaxInfluence.simulateCascade(graph, influenceSet.union([vertexInd]))) 
                    
            infSums /= numRuns    
            influenceSet.add(numpy.argmax(infSums))
            influenceList.append(numpy.argmax(infSums))
            unvisited.remove(numpy.argmax(infSums))
    
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
            
                        

    def celf(self, graph ): 
        """
        Maximising the influence using the CELF algorithm of Leskovec et al. 
        """
        pass 
        