
"""
Wrap a single static dataset which each line is of the form userId itemId rating. 
"""
import gc 
import array
import numpy 
import logging
import scipy.sparse 
from exp.util.IdIndexer import IdIndexer

class Static2IdValDataset(object):
    def __init__(self, dataFilename, split=0.8):
        """
        Read datasets from the specified files.
        """
        printStep = 1000000        
        
        authorIndexer = IdIndexer() 
        itemIndexer = IdIndexer() 
        ratings = array.array("i")
        
        #Read train files 
        dataFile = open(dataFilename)
        for i, line in enumerate(dataFile): 
            if i % printStep == 0: 
                logging.debug("Iteration: " + str(i))
            vals = line.split() 
            
            authorIndexer.append(vals[0])
            itemIndexer.append(vals[1])
            ratings.append(int(vals[2]))
            
        dataFile.close()
        logging.debug("Read file with " + str(i+1) + " lines")
            
        authorInds = numpy.array(authorIndexer.getArray())
        itemInds = numpy.array(itemIndexer.getArray())
        ratings = numpy.array(ratings)
        
        logging.debug("Number of authors: " + str(len(authorIndexer.getIdDict())))
        logging.debug("Number of items: " + str(len(itemIndexer.getIdDict())))
        logging.debug("Number of ratings: " + str(ratings.shape[0]))
        
        del authorIndexer 
        del itemIndexer
        gc.collect()
        
        shape = (numpy.max(authorInds)+1, numpy.max(itemInds)+1)
        inds = numpy.random.permutation(ratings.shape[0])
        trainInds = inds[0:int(inds.shape[0]*split)]
        trainX = scipy.sparse.csc_matrix((ratings[trainInds], (authorInds[trainInds], itemInds[trainInds])), shape=shape)
        
        testInds = inds[int(inds.shape[0]*split):]
        testX = scipy.sparse.csc_matrix((ratings[testInds], (authorInds[testInds], itemInds[testInds])), shape=shape)
        
        del authorInds, itemInds, ratings 
        gc.collect()
        
        self.trainXList = [trainX]
        self.testXList = [testX]

    def getTrainIteratorFunc(self):
        return iter(self.trainXList)

    def getTestIteratorFunc(self):
        return iter(self.testXList)

