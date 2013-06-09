"""
Process the groups in Mendeley, since they give an indication of time variation. 
"""
import gc 
import os 
import array 
import numpy 
import logging
import scipy.sparse 
import time 
import pickle
from datetime import datetime 
from exp.util.SparseUtils import SparseUtils 
from exp.util.SparseUtilsCython import SparseUtilsCython
from apgl.util.PathDefaults import PathDefaults 
from apgl.util.Util import Util 

class MendeleyGroupsDataset(object): 
    def __init__(self, ): 
        outputDir = PathDefaults.getOutputDir() + "recommend/erasm/"

        if not os.path.exists(outputDir): 
            os.mkdir(outputDir)
                
        self.ratingFileName = outputDir + "data.npz"          
        self.userDictFileName = outputDir + "userIdDict.pkl"   
        self.groupDictFileName = outputDir + "groupIdDict.pkl" 
        self.isTrainRatingsFileName = outputDir + "is_train.npz"
    
        self.dataDir = PathDefaults.getDataDir() + "erasm/"
        self.dataFileName = self.dataDir + "groupMembers-29-11-12" 
        
        self.trainSplit = 4.0/5 
        
        self.processRatings()
        self.splitDataset()        
        self.loadProcessedData()
        
    def processRatings(self):
        """
        Take the data file and save it in a format for easier work in 
        
        """    
        if not os.path.exists(self.ratingFileName) or not os.path.exists(self.userDictFileName): 
            logging.debug("Processing ratings given in " + self.dataDir)

            userIdDict = {} 
            userIdSet = set([])    
            
            groupIdDict = {} 
            groupIdSet = set([])
            
            groupInds = array.array("I")
            userInds = array.array("I")
            dates = array.array("L")
            i = 0            
            j = 0
            
            itr = 0 
            ratingsFile = open(self.dataFileName)
            ratingsFile.readline()
            
            for line in ratingsFile: 
                if itr % 1000 == 0:
                    print(itr)
                #Util.printIteration(itr, 100000, self.numRatings)
                vals = line.split()
                
                userId = int(vals[1])
                
                if userId not in userIdSet: 
                    userIdSet.add(userId)
                    userIdDict[userId] = j
                    userInd = j 
                    j += 1 
                else: 
                    userInd = userIdDict[userId]
                    
                groupId = int(vals[0])
                
                if groupId not in groupIdSet: 
                    groupIdSet.add(groupId)
                    groupIdDict[groupId] = i
                    groupInd = i 
                    i += 1 
                else: 
                    groupInd = groupIdDict[groupId]
                     
                t = datetime.strptime(vals[3].strip(), "%Y-%m-%d")
            
                groupInds.append(groupInd)
                userInds.append(userInd)   
                dates.append(int(time.mktime(t.timetuple()))) 
                itr += 1 
                    
            groupInds = numpy.array(groupInds, numpy.uint32)
            userInds = numpy.array(userInds, numpy.uint32)
            dates = numpy.array(dates, numpy.uint32)
            
            X = scipy.sparse.csc_matrix((numpy.ones(userInds.shape[0]), (userInds, groupInds)))
            
            numpy.savez(self.ratingFileName, groupInds, userInds, dates) 
            logging.debug("Saved ratings file as " + self.ratingFileName)
            
            pickle.dump(userIdDict, open(self.userDictFileName, 'wb'))
            logging.debug("Saved userIdDict as " + self.userDictFileName)
            
            pickle.dump(groupIdDict, open(self.groupDictFileName, 'wb'))
            logging.debug("Saved groupIdDict as " + self.groupDictFileName)
        else: 
            logging.debug("Ratings file " + str(self.ratingFileName) + " already processed")


    def splitDataset(self): 
        """
        We generate a random training and test sets based on a specified split. 
        """
        if not os.path.exists(self.isTrainRatingsFileName):
            numpy.random.seed(21)
            custIdDict = pickle.load(open(self.userDictFileName))             
            dataArr = numpy.load(self.ratingFileName)
            groupInds, userInds, dates = dataArr["arr_0"], dataArr["arr_1"], dataArr["arr_2"]
            logging.debug("Number of ratings: " + str(userInds.shape[0]))            
            del dates 
            logging.debug("Training data loaded")
            
                       
            isTrainRating = numpy.array(numpy.random.rand(groupInds.shape[0]) <= self.trainSplit, numpy.bool)

            numpy.savez(self.isTrainRatingsFileName, isTrainRating) 
            logging.debug("Saved file as " + self.isTrainRatingsFileName)
        else: 
            logging.debug("Train/test indicators file " + str(self.isTrainRatingsFileName) + " already generated")

    def loadProcessedData(self): 
        dataArr = numpy.load(self.ratingFileName)
        groupInds, userInds, self.dates = dataArr["arr_0"], dataArr["arr_1"], dataArr["arr_2"]
        self.trainInds = numpy.c_[groupInds, userInds].T
        del groupInds
        del userInds
        self.startTimeStamp = numpy.min(self.dates)
        self.endTimeStamp = numpy.max(self.dates)
        logging.debug("Training data loaded")
        logging.debug("Number of ratings: " + str(self.trainInds.shape[0]+1))
        
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

    def bipartiteToUni(self, userInds, itemInds, dates): 
        """
        Take a bipartite graph X with rows as users and columns as items 
        and create a graph of similar users (i.e. those who like the same 
        item). 
        """
        inds = numpy.argsort(itemInds)
        
        userInds = userInds[inds]
        itemInds = itemInds[inds]
        dates = dates[inds]
        
        users1 = array.array("I")
        users2 = array.array("I")
        newDates = array.array("L")
        
        lastItem = -1
        
        for user, item, date in zip(userInds, itemInds, dates): 
            if lastItem != item: 
                for u1, d1 in currentUserList: 
                    for u2, d2 in currentUserList: 
                        if u1 != u2: 
                            users1.add(u1)
                            users2.add(u2)
                            newDates.add(numpy.max(d1, d2))
                
                currentUserList = []
            else: 
                currentUserList.append((user, date)) 
            
            lastItem = item
    
        user1 = numpy.array(users1, numpy.uint32)
        user2 = numpy.array(users2, numpy.uint32)
        newDates = numpy.array(newDates, numpy.uint32)
        
        return user1, user2, newDates
        

numpy.set_printoptions(suppress=True, precision=3)
dataset = MendeleyGroupsDataset()
dataset.processRatings()
dataset.splitDataset()
