import numpy
import array  
import logging 
import sys 
import pickle 
import os 
import scipy.sparse 
from datetime import datetime, timedelta   
from exp.util.SparseUtils import SparseUtils 
from apgl.util.PathDefaults import PathDefaults 
from apgl.util.Util import Util 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class NetflixDataset(object): 
    def __init__(self): 
        """
        Return a training and test set for netflix based on the time each 
        rating was made. 
        """ 
        self.timeStep = 30 
        self.startDate = datetime(1998,1,1)
        
        self.startMovieID = 1 
        self.endMovieID = 17770
        
        outputDir = PathDefaults.getOutputDir() + "recommend/netflix/"
        self.ratingFileName = outputDir + "data.npz"  
        self.custDictFileName = outputDir + "custIdDict.pkl"
        self.probeFileName = PathDefaults.getDataDir() + "netflix/probe.txt"    
        self.testRatingsFileName = outputDir + "test_data.npz"
        
        #self.processRatings() 
        #self.processProbe() 
    
    def processRatings(self): 
        """
        Convert the dataset into a matrix and save the results for faster 
        access. 
        """
        if not os.path.exists(self.ratingFileName) or not os.path.exists(self.custDictFileName): 
            dataDir = PathDefaults.getDataDir() + "netflix/training_set/"

            custIdDict = {} 
            custIdSet = set([])        
            
            movieIds = array.array("I")
            custIds = array.array("I")
            ratings = array.array("B")
            dates = array.array("L")
            j = 0
            
            for i in range(self.startMovieID, self.endMovieID): 
                Util.printIteration(i-1, 1, self.endMovieID-1)
                ratingsFile = open(dataDir + "mv_" + str(i).zfill(7) + ".txt")
                ratingsFile.readline()
                
                for line in ratingsFile: 
                    vals = line.split(",")
                    
                    custId = int(vals[0])
                    
                    if custId not in custIdSet: 
                        custIdSet.add(custId)
                        custIdDict[custId] = j
                        custInd = j 
                        j += 1 
                    else: 
                        custInd = custIdDict[custId]
                    
                    rating = int(vals[1])     
                    t = datetime.strptime(vals[2].strip(), "%Y-%m-%d")
                
                    movieIds.append(i-1)
                    custIds.append(custInd)   
                    ratings.append(rating)
                    dates.append(int((t-self.startDate).total_seconds()))
                    
            movieIds = numpy.array(movieIds, numpy.uint32)
            custIds = numpy.array(custIds, numpy.uint32)
            ratings = numpy.array(ratings, numpy.uint8)
            dates = numpy.array(dates)
            
            numpy.savez(self.ratingFileName, movieIds, custIds, ratings, dates) 
            logging.debug("Saved ratings file as " + self.ratingFileName)
            
            pickle.dump(custIdDict, open(self.custDictFileName, 'wb'))
            logging.debug("Saved custIdDict as " + self.custDictFileName)
        else: 
            logging.debug("Ratings file " + str(self.ratingFileName) + " already processed")

    def processProbe(self): 
        """
        We go through the probe set and create a boolean array over the dataset 
        to indicate whether a rating is part of the training or test set. 
        """
        if True or not os.path.exists(self.testRatingsFileName):
            movieIds = array.array("I")
            custIds = array.array("I")
            custIdDict = pickle.load(open(self.custDictFileName))         
            
            probeFile = open(self.probeFileName)
            i = 0 
            
            for line in probeFile: 
                if line.find(":") != -1: 
                    Util.printIteration(i, 100, self.endMovieID-1)
                    movieId = line[0:-2]
                    movieInd = int(movieId)-1
                    i += 1
                else: 
                    custId = int(line.strip())
                    custInd = custIdDict[custId]
                    
                    movieIds.append(movieInd)
                    custIds.append(custInd)  
                    
            movieIds = numpy.array(movieIds, numpy.uint32)
            custIds = numpy.array(custIds, numpy.uint32)   
        
            numpy.savez(self.testRatingsFileName, movieIds, custIds) 
            logging.debug("Saved file as " + self.testRatingsFileName)
        else: 
            logging.debug("Test ratings file " + str(self.testRatingsFileName) + " already processed")                

    def getTrainIteratorFunc(self): 
        class NetflixIterator(object): 
            def __init__(self, netflixDataset): 
                self.currentDate = datetime(2000,1,1)
                self.timeDelta = timedelta(netflixDataset.timeStep)
                self.netflixDataset = netflixDataset
                
                dataArr = numpy.load(netflixDataset.ratingFileName)
                movieIds, custIds, self.ratings, self.dates = dataArr["arr_0"], dataArr["arr_1"], dataArr["arr_2"], dataArr["arr_3"]
                self.trainInds = numpy.c_[movieIds, custIds].T
                logging.debug("Training data loaded")
                logging.debug("Number of ratings: " + str(self.ratings.shape[0]+1))
                
                logging.debug("Sorting dates")
                self.dateInds = numpy.argsort(self.dates)
                self.sortedDates = self.dates[self.dateInds]
                
                dataArr = numpy.load(netflixDataset.testRatingsFileName)
                testMovieIds, testCustIds= dataArr["arr_0"], dataArr["arr_1"]
                self.testInds = numpy.c_[testMovieIds, testCustIds].T
                logging.debug("Test data loaded")
                
            def next(self): 
                timeInt = int((self.currentDate-self.netflixDataset.startDate).total_seconds())    
                ind = numpy.searchsorted(self.sortedDates, timeInt)
                
                currentRatings = self.ratings[self.dateInds[0:ind]]
                currentInds = self.trainInds[:, self.dateInds[0:ind]]
                
                print(currentRatings.shape, currentInds.shape)
                trainX = scipy.sparse.csc_matrix((currentRatings, currentInds))
                
                self.currentDate += self.timeDelta
                
                return trainX
                
                
        return NetflixIterator(self)
                
                
              
dataset = NetflixDataset()
#dataset.processRatings()
dataset.processProbe()

iterator = dataset.getTrainIteratorFunc()

X = iterator.next() 
print(X.shape, X.nnz)

