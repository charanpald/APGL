import numpy 
import logging
from apgl.util.Util import Util 

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
    
    @staticmethod
    def generateItemList(lists): 
        itemList = set([])

        for lst in lists: 
            itemList = itemList.union(set(lst))
        
        itemList = list(itemList)
        return itemList 
    
    @staticmethod 
    def MC2(lists, itemList, alpha=None, verbose=False): 
        """
        Perform weighted rank aggregation using MC2 as given in Rank Aggregation Methods 
        for the Web, Dwork et al. The weighting vector is given by alpha. 
        
        :param lists: A list of lists. Each sublist is an ordered set of a subset of the items from itemList 
        
        :param itemList: A list of all possible items 
        """
        
        n = len(itemList)
        ell = len(lists)
        
        if alpha == None: 
            alpha = numpy.ones(ell)/ell
        
        P = numpy.zeros((n, n))
        PList = [] 
        
        for j, lst in enumerate(lists): 
            Util.printIteration(j, 1, ell)
            Pj = numpy.zeros((n, n))
            
            indexList = numpy.zeros(len(lst), numpy.int)            
            
            for i, item in enumerate(lst): 
                indexList[i] = itemList.index(item)
                
            for i in range(indexList.shape[0]): 
                validStates = indexList[0:i+1]
                Pj[indexList[i], validStates] = 1.0/validStates.shape[0]

            P += alpha[j] * Pj 
            PList.append(Pj)
        
        P /= ell 

        #If all lists agree on top elements then we get a stationary distribution 
        #of 1 for that index and zero elsewhere. Therefore add a little noise. 
        P += numpy.ones((n, n))*0.0001
        for i in range(n): 
            P[i, :] = P[i, :]/P[i, :].sum()
                
        #logging.debug("Computing eigen-decomposition of P with shape" + str(P.shape))
        #u, V = numpy.linalg.eig(P.T)
        #scores = numpy.abs(V[:, 0])

        u, v = Util.powerEigs(P.T, 0.001)
        scores = numpy.abs(v)
        assert abs(u-1) < 0.001

        inds = numpy.flipud(numpy.argsort(scores)) 
        
        outputList = [] 
        for ind in inds: 
            outputList.append(itemList[ind])
        
        if verbose: 
            return outputList, scores, PList
        else: 
            return outputList, scores
              
    @staticmethod    
    def supervisedMC2(lists, itemList, topQList): 
        """
        Use the supervised rank aggregation from the paper "Supervised Rank 
        Aggregation". 
        """
        outputList, scores, PList = RankAggregator.MC2(lists, itemList, verbose=True)
        ell = len(PList)
        n = len(itemList)        
        q = len(topQList)
        
        A = numpy.zeros((ell, n))
        for i in range(ell): 
            A[i, :] = numpy.diag(PList[i])
        
        notQList = list(set(itemList).difference(set(topQList)))    
        
        #Construct matrix H 
        H = numpy.zeros((q*(n-q), n))
        k = 0        
        
        for i in range(q): 
            for j in range(n-q): 
                H[itemList.indexof(topQList[i]), k] = -1 
                H[itemList.indexof(notQList[j]), k] = 1 
                k += 1 
        #Might need a way to limit the size of H 
        r = H.shape[0]
        
        zeroEllEll = numpy.zeros((ell, ell))
        zeroEllR =numpy.zeros((ell, ell)) 
        zeroNN = numpy.zeros((n, n))
        H00 = numpy.c_[zeroEllEll, -A, zeroEllR]
        H01 = numpy.c_[-A.T, zeroNN, numpy.zeros((n, r))]
        H02 = numpy.c_[numpy.zeros((m, ell)), numpy.zeros((r, n)), numpy.eye(r)]
        H0 = numpy.r_[H00, H01, H02]
        
        H10 = numpy.c_[-numpy.eye(ell), numpy.zeros((ell, n)), numpy.zeros((ell, r))]
        H11 = numpy.c_[-numpy.zeros((r, ell)), numpy.zeros((r, n)), numpy.eye(r)]
        H1 = numpy.r_[H10, H11]
        
        H2 = numpy.zeros((2, (ell+n+r)))
        H2[0, 0:ell]  = 1 
        H2[1, ell:(ell+n)] = 1
        
        H30 = numpy.c_[numpy.zeros((n, ell)), -numpy.eye(n), numpy.zeros((n, r))]
        H31 = numpy.c_[numpy.zeros((r, ell)), H, -numpy.eye(r)]
        H3 = numpy.r_[H30, H31]
        
        #We have vector rho = [lambda_0, lambda_1, lambda_2, lambda_3, gamma]
        c = numpy.zeros(ell+n+2*r+3)
        c[-1] = -1 
        
        Gl1 = numpy.zeros((ell+n+2*r+2, ell+n+2*r+3)) 
        numpy.put(Gl, numpy.arange(ell+n+2*r+2), numpy.arange(ell+n+2*r+2), 1)
        hl1 = numpy.zeros(ell+n+2*r+2)        
        
        #Finish later 
        #Gl2 = 
        
                
        
        
        
        
        
        
 