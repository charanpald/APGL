import numpy
import array  
import logging 
import sys 
import pickle 
import os 
import scipy.sparse 
import numpy.testing as nptst 
import gc 
import time 
from datetime import datetime, timedelta   
from apgl.util.PathDefaults import PathDefaults 
from apgl.util.Util import Util 
from exp.recommendexp.TimeStamptedIterator import TimeStamptedIterator

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)  

class EpinionsDataset(object): 
    def __init__(self, maxIter=None, iterStartTimeStamp=None): 
        """
        Return a training and test set for itemlens based on the time each 
        rating was made. 
        """ 
        self.timeStep = timedelta(30).total_seconds() 
        
        #iterStartDate is the starting date of the iterator 
        if iterStartTimeStamp != None: 
            self.iterStartTimeStamp = iterStartTimeStamp
        else: 
            self.iterStartTimeStamp = time.mktime(datetime(2002,1,1).timetuple())
         
        self.numItems = 1560144
        #It says 13668319 on the site but that seems to be wrong 
        self.numRatings = 13668320
        self.numCustomers = 71567
        
        outputDir = PathDefaults.getOutputDir() + "recommend/epinions/"

        if not os.path.exists(outputDir): 
            os.mkdir(outputDir)
                
        self.ratingFileName = outputDir + "data.npz"  
        self.custDictFileName = outputDir + "custIdDict.pkl"   
        self.itemDictFileName = outputDir + "itemIdDict.pkl" 
        self.isTrainRatingsFileName = outputDir + "is_train.npz"
        
        self.maxIter = maxIter 
        self.trainSplit = 4.0/5 

        self.processRatings()
        self.splitDataset()        
        self.loadProcessedData()
        
        if self.maxIter != None: 
            logging.debug("Maximum number of iterations: " + str(self.maxIter))

    def processRatings(self): 
        """
        Convert the dataset into a matrix and save the results for faster 
        access. 
        """
        if not os.path.exists(self.ratingFileName) or not os.path.exists(self.custDictFileName): 
            dataDir = PathDefaults.getDataDir() + "epinions/"

            logging.debug("Processing ratings given in " + dataDir)

            custIdDict = {} 
            custIdSet = set([])    
            
            itemIdDict = {} 
            itemIdSet = set([])
            
            itemInds = array.array("I")
            custInds = array.array("I")
            ratings = array.array("f")
            dates = array.array("L")
            i = 0            
            j = 0
            
            itr = 0 
            ratingsFile = open(dataDir + "rating.txt")
            
            for line in ratingsFile: 
                Util.printIteration(itr, 100000, self.numRatings)
                vals = line.split()
                
                custId = int(vals[1])
                
                if custId not in custIdSet: 
                    custIdSet.add(custId)
                    custIdDict[custId] = j
                    custInd = j 
                    j += 1 
                else: 
                    custInd = custIdDict[custId]
                    
                itemId = int(vals[0])
                
                if itemId not in itemIdSet: 
                    itemIdSet.add(itemId)
                    itemIdDict[itemId] = i
                    itemInd = i 
                    i += 1 
                else: 
                    itemInd = itemIdDict[itemId]
                    
                rating = float(vals[2])
                
                #Sometimes there is a modification date, so we use that 
                if len(vals) == 7:
                    t = datetime.strptime(vals[4].strip(), "%Y/%m/%d")
                else: 
                    t = datetime.strptime(vals[5].strip(), "%Y/%m/%d")
                    
                t = int(time.mktime(t.timetuple()))                    
                    
                itemInds.append(itemInd)
                custInds.append(custInd)   
                ratings.append(rating)
                dates.append(t)
                itr += 1 
                    
            itemInds = numpy.array(itemInds, numpy.uint32)
            custInds = numpy.array(custInds, numpy.uint32)
            ratings = numpy.array(ratings, numpy.float)
            dates = numpy.array(dates, numpy.uint32)
            
            print(ratings.shape[0])
            assert ratings.shape[0] == self.numRatings            
            
            numpy.savez(self.ratingFileName, itemInds, custInds, ratings, dates) 
            logging.debug("Saved ratings file as " + self.ratingFileName)
            
            pickle.dump(custIdDict, open(self.custDictFileName, 'wb'))
            logging.debug("Saved custIdDict as " + self.custDictFileName)
            
            pickle.dump(itemIdDict, open(self.itemDictFileName, 'wb'))
            logging.debug("Saved itemIdDict as " + self.itemDictFileName)
        else: 
            logging.debug("Ratings file " + str(self.ratingFileName) + " already processed")
    
    def splitDataset(self): 
        """
        We generate a random training and test sets based on a specified split. 
        """
        if not os.path.exists(self.isTrainRatingsFileName):
            numpy.random.seed(21)
            custIdDict = pickle.load(open(self.custDictFileName))             
            dataArr = numpy.load(self.ratingFileName)
            itemInds, custInds, ratings, dates = dataArr["arr_0"], dataArr["arr_1"], dataArr["arr_2"], dataArr["arr_3"]
            logging.debug("Number of ratings: " + str(ratings.shape[0]))            
            del ratings, dates 
            logging.debug("Training data loaded")
            
                       
            isTrainRating = numpy.array(numpy.random.rand(itemInds.shape[0]) <= self.trainSplit, numpy.bool)

            numpy.savez(self.isTrainRatingsFileName, isTrainRating) 
            logging.debug("Saved file as " + self.isTrainRatingsFileName)
        else: 
            logging.debug("Train/test indicators file " + str(self.isTrainRatingsFileName) + " already generated")
        
    def loadProcessedData(self): 
        dataArr = numpy.load(self.ratingFileName)
        itemInds, custInds, self.ratings, self.dates = dataArr["arr_0"], dataArr["arr_1"], dataArr["arr_2"], dataArr["arr_3"]
        self.trainInds = numpy.c_[itemInds, custInds].T
        del itemInds
        del custInds
        self.startTimeStamp = numpy.min(self.dates)
        self.endTimeStamp = numpy.max(self.dates)
        logging.debug("Training data loaded")
        logging.debug("Number of ratings: " + str(self.ratings.shape[0]+1))
                
        self.isTrainRating = numpy.load(self.isTrainRatingsFileName)["arr_0"]
        logging.debug("Train/test indicator loaded")              
     
        logging.debug("Sorting dates")
        self.dateInds = numpy.array(numpy.argsort(self.dates), numpy.uint32)
        self.sortedDates = self.dates[self.dateInds]
        logging.debug("Done")
        gc.collect()
           
    def getTrainIteratorFunc(self): 
        return TimeStamptedIterator(self, True)
                
    def getTestIteratorFunc(self): 
        return TimeStamptedIterator(self, False)           
              