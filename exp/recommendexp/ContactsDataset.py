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
from exp.util.SparseUtils import SparseUtils 
from apgl.util.PathDefaults import PathDefaults 
from apgl.util.Util import Util 
from exp.recommendexp.TimeStamptedIterator import TimeStamptedIterator

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)  

class ContactsDataset(object): 
    def __init__(self, maxIter=None, iterStartTimeStamp=None): 
        """
        Return a training and test set for movielens based on the time each 
        rating was made. 
        """ 
        self.timeStep = timedelta(30).total_seconds() 
        
        #iterStartDate is the starting date of the iterator 
        if iterStartTimeStamp != None: 
            self.iterStartTimeStamp = iterStartTimeStamp
        else: 
            self.iterStartTimeStamp = 789652009
         
        outputDir = PathDefaults.getOutputDir() + "recommend/erasm/"

        self.numRatings = 402872
        self.minContacts = 10 

        if not os.path.exists(outputDir): 
            os.mkdir(outputDir)
                
        self.ratingFileName = outputDir + "data.npz"  
        self.userDictFileName = outputDir + "userIdDict.pkl"   
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
        if not os.path.exists(self.ratingFileName) or not os.path.exists(self.userDictFileName): 
            dataDir = PathDefaults.getDataDir() + "erasm/"

            logging.debug("Processing ratings given in " + dataDir)

            userIdDict = {} 
            userIdSet = set([])    
            userNumContacts = []
            
            userInds1 = array.array("I")
            userInds2 = array.array("I")
            dates = array.array("L")       
            j = 0
            
            itr = 0 
            ratingsFile = open(dataDir + "connections-28-11-12")
            ratingsFile.readline()
            
            for line in ratingsFile: 
                Util.printIteration(itr, 1000, self.numRatings)
                vals = line.split()
                
                userId1 = int(vals[0])
                userId2 = int(vals[1])
                tempUserInds = []                
                
                for i, userId in enumerate([userId1, userId2]): 
                    if userId not in userIdSet: 
                        userIdSet.add(userId)
                        userIdDict[userId] = j
                        userNumContacts.append(1) 
                        userInd = j 
                        j += 1 
                    else: 
                        userInd = userIdDict[userId]
                        userNumContacts[userIdDict[userId]] += 1
                   
                    tempUserInds.append(userInd)
                    
                startDate = datetime(2000, 1, 1)
                endDate = datetime(2010, 12, 31)
                timestamp = int(time.mktime((startDate + timedelta(seconds=numpy.random.randint(0, int((endDate - startDate).total_seconds())))).timetuple())) 
            
                userInds1.append(tempUserInds[0])   
                userInds2.append(tempUserInds[1])
                dates.append(timestamp)
                itr += 1 
                  
            
            userInds1 = numpy.array(userInds1, numpy.uint32)
            userInds2 = numpy.array(userInds2, numpy.uint32)
            dates = numpy.array(dates, numpy.uint32)
            userNumContacts = numpy.array(userNumContacts, numpy.int)
            
            assert userInds1.shape[0] == self.numRatings     
            
            #Go through and find people with at least 10 contacts 
            userInds1 = userInds1[userNumContacts >= self.minContacts]
            userInds2 = userInds2[userNumContacts >= self.minContacts]
            dates = dates[userNumContacts >= self.minContacts]
            
            trainInds = numpy.c_[userInds1, userInds2].T
            logging.debug("Number of ratings after post-processing : " + str(trainInds.shape[1]))
            
            numpy.savez(self.ratingFileName, trainInds, dates) 
            logging.debug("Saved ratings file as " + self.ratingFileName)
            
            pickle.dump(userIdDict, open(self.userDictFileName, 'wb'))
            logging.debug("Saved userIdDict as " + self.userDictFileName)
            
        else: 
            logging.debug("Ratings file " + str(self.ratingFileName) + " already processed")
    
    def splitDataset(self): 
        """
        We generate a random training and test sets based on a specified split. 
        """
        if not os.path.exists(self.isTrainRatingsFileName):
            numpy.random.seed(21)
            userIdDict = pickle.load(open(self.userDictFileName))             
            dataArr = numpy.load(self.ratingFileName)
            trainInds, dates = dataArr["arr_0"], dataArr["arr_1"]
            logging.debug("Number of ratings: " + str(dates.shape[0]))            
            del trainInds 
            logging.debug("Training data loaded")
            
            isTrainRating = numpy.array(numpy.random.rand(dates.shape[0]) <= self.trainSplit, numpy.bool)

            numpy.savez(self.isTrainRatingsFileName, isTrainRating) 
            logging.debug("Saved file as " + self.isTrainRatingsFileName)
        else: 
            logging.debug("Train/test indicators file " + str(self.isTrainRatingsFileName) + " already generated")
        
    def loadProcessedData(self): 
        dataArr = numpy.load(self.ratingFileName)
        self.trainInds, self.dates = dataArr["arr_0"], dataArr["arr_1"]
        self.ratings = numpy.ones(self.dates.shape[0], numpy.uint8)
        self.startTimeStamp = numpy.min(self.dates)
        self.endTimeStamp = numpy.max(self.dates)
        logging.debug("Training data loaded")
        logging.debug("Number of ratings: " + str(self.dates.shape[0]))
        
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
              