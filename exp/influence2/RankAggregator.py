import numpy 

class RankAggregator(object): 
    def __init__(self): 
        pass 
    
    @staticmethod 
    def spearmanFootrule(list1, list2): 
        """
        Compute the spearman footrule distance between two ranked lists. The lists 
        must be the same size. 
        """
        dist = 0 
        
        score1 = numpy.zeros(len(list1))
        score2 = numpy.zeros(len(list2))
        
        score1[list1] = numpy.arange(len(list1))        
        score2[list2] = numpy.arange(len(list2))        
        
        for i in range(len(list1)): 
            dist += abs(score1[i] - score2[i])
            
        dist /= (len(list1)**2)/2
        
        return dist 


    @staticmethod 
    def borda(list1, list2): 
        """
        Borda's method for combining rankings. 
        """
        score1 = numpy.zeros(len(list1))
        score2 = numpy.zeros(len(list2))
        
        score1[list1] = numpy.flipud(numpy.arange(len(list1)))     
        score2[list2] = numpy.flipud(numpy.arange(len(list2)))    
        
        totalScore = score1 + score2 
        
        return numpy.flipud(numpy.argsort(totalScore))
        
    
