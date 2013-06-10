"""
Process the groups in Mendeley, since they give an indication of time variation. 
"""
import gc 
import os 
import array 
import numpy 
import logging
import time 
import pickle
import itertools 
from datetime import datetime, timedelta  
from apgl.util.PathDefaults import PathDefaults 
from apgl.util.Util import Util 
from exp.recommendexp.TimeStamptedIterator import TimeStamptedIterator


class MendeleyGroupsDataset(object): 
    def __init__(self, maxIter=None, iterStartTimeStamp=None): 
        outputDir = PathDefaults.getOutputDir() + "recommend/erasm/"

        if not os.path.exists(outputDir): 
            os.mkdir(outputDir)
            
        #iterStartDate is the starting date of the iterator 
        if iterStartTimeStamp != None: 
            self.iterStartTimeStamp = iterStartTimeStamp
        else: 
            self.iterStartTimeStamp = 1286229600
            
        self.timeStep = timedelta(30).total_seconds()             
                
        self.ratingFileName = outputDir + "data.npz"          
        self.userDictFileName = outputDir + "userIdDict.pkl"   
        self.groupDictFileName = outputDir + "groupIdDict.pkl" 
        self.isTrainRatingsFileName = outputDir + "is_train.npz"
    
        self.dataDir = PathDefaults.getDataDir() + "erasm/"
        self.dataFileName = self.dataDir + "groupMembers-29-11-12" 
        
        self.maxIter = maxIter 
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
                        
            logging.debug("Converting bipartite graph to unipartite")
            users1, users2, newDates = self.bipartiteToUni(userInds, groupInds, dates)
            
            numpy.savez(self.ratingFileName, users1, users2, newDates) 
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
        self.ratings = numpy.ones(groupInds.shape[0], numpy.bool)
        del groupInds
        del userInds
        self.startTimeStamp = numpy.min(self.dates)
        self.endTimeStamp = numpy.max(self.dates)
        logging.debug("Training data loaded")
        logging.debug("Number of ratings: " + str(self.trainInds.shape[1]+1))
        
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
        
        currentUserList = []
        i = 0
        
        X = {} 
        
        for user, item, date in zip(userInds, itemInds, dates): 
            if i % 1000 == 0:
                print(i)            
            
            if (i != userInds.shape[0]-1 and itemInds[i+1] != item) or i == userInds.shape[0]-1: 
                currentUserList.append((user, date)) 
                
                for ud1, ud2 in itertools.permutations(currentUserList, 2): 
                    u1, d1 = ud1 
                    u2, d2 = ud2
                    
                    if (u1, u2) not in X: 
                        X[(u1, u2)] = max(d1, d2)
                    else: 
                        X[(u1, u2)] = min(max(d1, d2), X[(u1, u2)])
                
                currentUserList = []
            else: 
                currentUserList.append((user, date))
            
            i += 1
            
        for i, j in X.keys(): 
            users1.append(i)
            users2.append(j)
            newDates.append(X[(i, j)])
        
        users1 = numpy.array(users1)
        users2 = numpy.array(users2)
        newDates = numpy.array(newDates)
        
        inds = numpy.argsort(users1)
        
        users1 = users1[inds]
        users2 = users2[inds]
        newDates = newDates[inds]

        return users1, users2, newDates
        