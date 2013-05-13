import numpy
import array  
import logging 
import sys 
import pickle 
import os 
import scipy.sparse 
import numpy.testing as nptst 
from datetime import datetime, timedelta   
from exp.util.SparseUtils import SparseUtils 
from apgl.util.PathDefaults import PathDefaults 
from apgl.util.Util import Util 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class NetflixIterator(object): 
    def __init__(self, netflixDataset, isTraining): 
        """
        Initialise this iterator with a NetflixDataset object and indicate whether 
        we want the training or test set. 
        """
        self.currentDate = datetime(2001,1,1)
        self.timeDelta = timedelta(netflixDataset.timeStep)
        self.netflixDataset = netflixDataset
        
        dataArr = numpy.load(netflixDataset.ratingFileName)
        movieIds, custIds, self.ratings, self.dates = dataArr["arr_0"], dataArr["arr_1"], dataArr["arr_2"], dataArr["arr_3"]
        self.trainInds = numpy.c_[movieIds, custIds].T
        logging.debug("Training data loaded")
        logging.debug("Number of ratings: " + str(self.ratings.shape[0]+1))
        
        isTrainRating = numpy.load(netflixDataset.isTrainRatingsFileName)["arr_0"]
        logging.debug("Train/test indicator loaded")                
     
        if isTraining:
            self.trainInds = self.trainInds[:, isTrainRating]
            self.ratings = self.ratings[isTrainRating]         
            self.dates = self.dates[isTrainRating]  
        else: 
            isTestRating = numpy.logical_not(isTrainRating)
            self.trainInds = self.trainInds[:, isTestRating]
            self.ratings = self.ratings[isTestRating]         
            self.dates = self.dates[isTestRating]  
            
        logging.debug("Final number of ratings " + str(self.ratings.shape[0])) 
     
        logging.debug("Sorting dates")
        self.dateInds = numpy.argsort(self.dates)
        self.sortedDates = self.dates[self.dateInds]
                        
    def next(self):
        logging.debug(self.currentDate)
        if self.currentDate > self.netflixDataset.endDate + self.timeDelta: 
            raise StopIteration
        
        timeInt = int((self.currentDate-self.netflixDataset.startDate).total_seconds())    
        ind = numpy.searchsorted(self.sortedDates, timeInt)
        
        currentRatings = self.ratings[self.dateInds[0:ind]]
        currentInds = self.trainInds[:, self.dateInds[0:ind]]
        
        X = scipy.sparse.csc_matrix((currentRatings, currentInds))                
        self.currentDate += self.timeDelta

        return X

    def __iter__(self):
        return self    

class NetflixDataset(object): 
    def __init__(self): 
        """
        Return a training and test set for netflix based on the time each 
        rating was made. 
        """ 
        self.timeStep = 30 
        self.startDate = datetime(1998,1,1)
        self.endDate = datetime(2005,12,31)
        
        self.startMovieID = 1 
        self.endMovieID = 17770
        
        self.numMovies = 17770
        self.numRatings = 100480507
        self.numProbeMovies = 16938
        self.numProbeRatings = 1408395
        self.numCustomers = 480189
        
        outputDir = PathDefaults.getOutputDir() + "recommend/netflix/"

        if not os.path.exists(outputDir): 
            os.mkdir(outputDir)
                
        self.ratingFileName = outputDir + "data.npz"  
        self.custDictFileName = outputDir + "custIdDict.pkl"
        self.probeFileName = PathDefaults.getDataDir() + "netflix/probe.txt"    
        self.testRatingsFileName = outputDir + "test_data.npz"
        self.isTrainRatingsFileName = outputDir + "is_train.npz"

        self.processRatings()
        self.processProbe()

    def processRatings(self): 
        """
        Convert the dataset into a matrix and save the results for faster 
        access. 
        """
        if not os.path.exists(self.ratingFileName) or not os.path.exists(self.custDictFileName): 
            dataDir = PathDefaults.getDataDir() + "netflix/training_set/"

            logging.debug("Processing ratings given in " + dataDir)

            custIdDict = {} 
            custIdSet = set([])        
            
            movieIds = array.array("I")
            custIds = array.array("I")
            ratings = array.array("B")
            dates = array.array("L")
            j = 0
            
            for i in range(self.startMovieID, self.endMovieID+1): 
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
            
            assert ratings.shape[0] == self.numRatings            
            
            numpy.savez(self.ratingFileName, movieIds, custIds, ratings, dates) 
            logging.debug("Saved ratings file as " + self.ratingFileName)
            
            pickle.dump(custIdDict, open(self.custDictFileName, 'wb'))
            logging.debug("Saved custIdDict as " + self.custDictFileName)
        else: 
            logging.debug("Ratings file " + str(self.ratingFileName) + " already processed")

    def processProbe(self): 
        """
        Go through the probe set and label the corresponding ratings in the full 
        dataset as test. 
        """
        if not os.path.exists(self.isTrainRatingsFileName):
            custIdDict = pickle.load(open(self.custDictFileName))             
            dataArr = numpy.load(self.ratingFileName)
            movieInds, custInds, ratings, dates = dataArr["arr_0"], dataArr["arr_1"], dataArr["arr_2"], dataArr["arr_3"]
            logging.debug("Number of ratings: " + str(ratings.shape[0]+1))            
            del ratings, dates 
            logging.debug("Training data loaded")
            
            isTrainRating = numpy.ones(movieInds.shape[0], numpy.bool)
            probeFile = open(self.probeFileName)
            i = 0 
            
            #First figure out the movie boundaries 
            movieBoundaries = numpy.nonzero(numpy.diff(movieInds) != 0)[0] + 1
            movieBoundaries = numpy.insert(movieBoundaries, 0, 0)
            movieBoundaries = numpy.append(movieBoundaries, movieInds.shape[0])
            
            assert movieBoundaries.shape[0] == self.numMovies+1 
            assert movieBoundaries[-1] == movieInds.shape[0]
            
            for line in probeFile: 
                if line.find(":") != -1: 
                    Util.printIteration(i, 10, self.numProbeMovies)
                    movieId = line[0:-2]
                    movieInd = int(movieId)-1
                
                    startInd = movieBoundaries[movieInd] 
                    endInd = movieBoundaries[movieInd+1] 
                    #All the customers that watches movie movieInd
                    tempCustInds = custInds[startInd:endInd]
                    sortedInds = numpy.argsort(tempCustInds)
                    
                    assert (movieInds[startInd:endInd] == movieInd).all()
                    
                    i += 1
                else: 
                    custId = int(line.strip())
                    custInd = custIdDict[custId]

                    offset = numpy.searchsorted(tempCustInds[sortedInds], custInd)
                    isTrainRating[startInd + sortedInds[offset]] = 0 
                    
                    assert custInds[startInd + sortedInds[offset]] == custInd
               
            assert i == self.numProbeMovies 
            assert numpy.logical_not(isTrainRating).sum() == self.numProbeRatings               
               
            numpy.savez(self.isTrainRatingsFileName, isTrainRating) 
            logging.debug("Saved file as " + self.isTrainRatingsFileName)
        else: 
            logging.debug("Train/test indicators file " + str(self.isTrainRatingsFileName) + " already processed")
            
            
    def getTrainIteratorFunc(self): 
        return NetflixIterator(self, True)
                
    def getTestIteratorFunc(self): 
        return NetflixIterator(self, False)           
              
dataset = NetflixDataset()
#iterator = dataset.getTrainIteratorFunc()
iterator = dataset.getTestIteratorFunc()

for X in iterator: 
    print(X.shape, X.nnz)

