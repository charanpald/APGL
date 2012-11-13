
"""
Some classes to iterative over graph sequences 
"""
import itertools
import numpy
import scipy
import random
from apgl.graph.DictGraph import DictGraph
import logging
import scipy.sparse 

class MyDictionary(object):
    def __init__(self):
        self.d = dict()

    def is_known(self, key):
        return key in self.d

    def index(self, key):
        if not self.is_known(key):
            self.d[key] = len(self.d)
        return self.d.get(key)

    def __len__(self):
        return len(self.d)


class DatedPurchasesGroupByIterator(object):
    def __init__(self, purchasesByWeek, nb_purchases_per_it=None):
        """
        Take a list of purchases (sorted by week of purchase) and iterate on it.
        An iteration return a list of purchases done the same week.
        Purchases are given in a list of [user, prod, week, year] with increasing
        date.
        nb_purchases_per_it is the maximum number of purchases to put in each
        week (if there is more, split the week). None corresponds to
        no-limit case.
        """
        # args
        self.purchasesByWeek = purchasesByWeek
        self.nb_purchases_per_it = nb_purchases_per_it

        # init variables
        self.i_purchase = 0
        self.current_week_purchases = []

    def __iter__(self):
         return self

    def fill_current_week_purchases(self):
        self.current_week_purchases = []
        if self.i_purchase >= len(self.purchasesByWeek):
            return
        # current date: we skip weeks without purchases
        date = self.purchasesByWeek[self.i_purchase][2:]

        # read current week purchases
        while self.i_purchase<len(self.purchasesByWeek) and self.purchasesByWeek[self.i_purchase][2:] == date:
            self.current_week_purchases.append(self.purchasesByWeek[self.i_purchase])
            self.i_purchase += 1

        # shuffle purchases: abandonned as it would lead to different data for
        # each method, and it would lead to different data for learner and scorer
#        random.shuffle(self.current_week_purchases)
        

    def __next__(self):
        """
        Take first $self.nb_purchases_per_it$ purchases from
        $self.current_week_purchases$.
        If $self.current_week_purchases$ is empty, fill it with next week.
        """
        if not self.current_week_purchases: # if empty()
            self.fill_current_week_purchases()
            if not self.current_week_purchases: # if still empty()
                raise StopIteration
        
        returned_purchases = self.current_week_purchases[:self.nb_purchases_per_it]
        del self.current_week_purchases[:self.nb_purchases_per_it]
        return returned_purchases
        
    next = __next__ 


class DatedPurchasesGraphListIterator(object):
    def __init__(self, purchasesByWeek, nb_purchases_per_it=None):
        """
        The background graph is a bi-partite graph of purchases. Purchases are
        grouped by date (week by week), and we consider the graph of purchases
        between first week and $i$-th week.
        The returned graph considers only users and counts the number of common
        purchases between two users.
        Purchases are given in a list of [user, prod, week, year] with increasing
        date.
        nb_purchases_per_it is the maximum number of purchases to put in each
        week (if there is more, randomly split the week). None corresponds to
        no-limit case.
        """
        # args
        self.group_by_iterator = DatedPurchasesGroupByIterator(purchasesByWeek, nb_purchases_per_it)

        # init variables
        self.dictUser = MyDictionary()
        self.dictProd = MyDictionary()
        for user, prod, week, year in purchasesByWeek:
            self.dictUser.index(user)
            self.dictProd.index(prod)
        self.backgroundGraph = DictGraph(False) # directed
        self.W = scipy.sparse.csr_matrix((len(self.dictUser), len(self.dictUser)), dtype='int16')
        self.usefullEdges = numpy.array([])

    def __iter__(self):
         return self

    def __next__(self):
        # next group of purchases (StopIteration is raised here)
        purchases_sublist = next(self.group_by_iterator)
        #logging.debug(" nb purchases: " + str(len(purchases_sublist)))

        # to check that the group really induces new edges
        W_has_changed = False

        # update graphs adding current-week purchases
        for user, prod, week, year in purchases_sublist:
            # update only if this purchase is seen for the first time
            try:
                newEdge = not self.backgroundGraph.getEdge(prod, user)
            except ValueError:
                newEdge = True
            if newEdge:
                self.backgroundGraph.addEdge(prod, user)
                newCommonPurchases = self.backgroundGraph.neighbours(prod)
#                print prod, newCommonPurchases
                for neighbour in filter(lambda neighbour: neighbour != user, newCommonPurchases):
                    W_has_changed = True
                    self.W[neighbour, user] += 1
                    self.W[user, neighbour] += 1
        
        # the returned graph will be restricted to usefull edges
        currentUsefullEdges = numpy.array(self.W.sum(1)).ravel().nonzero()[0]
        newUsefullEdges = numpy.setdiff1d(currentUsefullEdges, self.usefullEdges)
        self.usefullEdges = numpy.r_[self.usefullEdges, newUsefullEdges]
        
        if W_has_changed:
          return self.W[self.usefullEdges,:][:,self.usefullEdges]
        else:
          return next(self)
    next = __next__ 



class IncreasingSubgraphListIterator(object):
    def __init__(self, graph, subgraphIndicesList):
        """
        the $i$-th returned graph is the subgraph of $graph$ using only indices
        $subgraphIndicesList[i]$. Each element of subgraphIndicesList must be a
        subset of the next.
        """
        # args
        self.W = graph.getSparseWeightMatrix()
        self.subgraphIndicesList = subgraphIndicesList

        # init variables
        self.i = 0
        self.subgraphIndices = numpy.array([], numpy.int)

    def __iter__(self):
         return self

    def __next__(self):
        if self.i >= len(self.subgraphIndicesList):
            raise StopIteration
        if self.i == 0:
            self.subgraphIndices = numpy.array(self.subgraphIndicesList[0], numpy.int)
            self.subgraphIndices.sort()
        else:
            if len(self.subgraphIndicesList[self.i]) >= len(self.subgraphIndices) :
                newSubgraphIndices = numpy.setdiff1d(numpy.array(self.subgraphIndicesList[self.i], numpy.int), self.subgraphIndices)
                self.subgraphIndices = numpy.r_[self.subgraphIndices, newSubgraphIndices]
            else:
                # check if removed indices are the last one
                deletedSubgraphIndices = numpy.setdiff1d(self.subgraphIndices, numpy.array(self.subgraphIndicesList[self.i], numpy.int))
                lastSubgraphIndices = self.subgraphIndices[len(self.subgraphIndicesList[self.i]):]
                deletedSubgraphIndices.sort()
                lastSubgraphIndices.sort()
                if not (deletedSubgraphIndices == lastSubgraphIndices).all():
                    logging.warn(" removing vertices which are not the last one")
                self.subgraphIndices = self.subgraphIndices[:len(self.subgraphIndicesList[self.i])]
                #print len(self.subgraphIndices)
        subW = self.W[:, self.subgraphIndices][self.subgraphIndices, :]
        self.i+=1
        return subW
    
    next = __next__ 


class toDenseGraphListIterator(object):
    """
    """

    def __init__(self, graphListIterator):
        self.g = graphListIterator

    def __iter__(self):
        return self

    def __next__(self):
        return numpy.array(next(self.g).todense(), numpy.float)
        
    next = __next__ 


class toSparseGraphListIterator(object):
    """
    """

    def __init__(self, graphListIterator):
        self.g = graphListIterator

    def __iter__(self):
        return self

    def __next__(self):
        return scipy.sparse.csr_matrix(next(self.g))
        
    next = __next__ 