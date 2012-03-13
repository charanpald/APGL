'''
Created on 8 May 2009

@author: charanpal

A class which models a set of set of examples used for a machine learning process.
An example refers to a particular row of all "data fields". A sub-data field is a set of rows 
of a particular data field. 
If a row vector is added as a data field, it is converted to a column vector (2D array). 
'''
#TODO: Figure out use cases 

from random import sample
from numpy import ix_, array, zeros, loadtxt
from numpy.random import permutation
import logging 
import scipy.io as io

class ExamplesList:
    def __init__(self, numExamples):
        self.__numExamples = int(numExamples)
        
        if numExamples < 0: 
            raise ValueError("Number of examples (" + str(numExamples) + ") must be greater than or equal to zero\n")

        self.__exampleIndices = array(list(range(0, numExamples)))
        self.__labelsName = 0
        self.__defaultExamplesName = 0
        self.__examples = {}
        logging.debug('Created ExamplesList with ' + str(numExamples) + ' examples')

    
    def getNumExamples(self):
        """Returns the number of examples in this object"""
        return self.__numExamples

    
    def addDataField(self, name, value):
        """Add an array of examples"""
        if name in self.__examples: 
            raise ValueError("Field already exists: " + name)
        
        self.__storeDataField(name, value)
       
    def __storeDataField(self, name, value):
        """ Store a data field without checking whether the field exists """  
        if value.shape[0] != self.__numExamples: 
            raise ValueError("Added data with incorrect number of examples")
        
        if value.ndim == 2:
            self.__examples[name] = value
        elif value.ndim == 1: #A row vector 
            self.__examples[name] = array([value]).T
        else: 
            raise ValueError("Invalid data field dimensionality: " + str(value.ndim))

    def overwriteDataField(self, name, value):
        """ Overwrite an existing data field with new examples """ 
        if not name in self.__examples: 
            raise ValueError("Cannot overwrite a field that does not exist: " + name)
        
        self.__storeDataField(name, value)

    def deleteDataField(self, name):
        if name not in self.__examples: 
            raise ValueError("Field does not exist: " + name)
        
        del self.__examples[name]

    def getNumberOfDataFields(self):
        return len(list(self.__examples.keys()))

    def deleteAllDataFields(self):
        self.__examples = {}

    def getDataFieldSize(self, name, dimension):
        if name not in self.__examples: 
            raise ValueError("Field does not exist: " + name)
        
        if dimension >= self.__examples[name].ndim or dimension < 0:
            raise ValueError("Invalid dimension: " + str(dimension))
        
        return self.__examples[name].shape[dimension]

    def getDataFieldNames(self):
        return list(self.__examples.keys())

    def getDataField(self, name):
        """
        Returns the data field according to the original ordering of data 
        """ 
        if name not in self.__examples: 
            raise ValueError("Field does not exist: " + name)
        
        return self.__examples[name]

    def getSampledDataField(self, name):
        """
        Returns the data field according to the current permutation or sampling of data. 
        """ 
        if name not in self.__examples: 
            raise ValueError("Field does not exist: " + name)
        
        if self.__exampleIndices.shape[0] != 0: 
            return (self.__examples[name])[self.__exampleIndices, :]
        else:
            return zeros((0, self.__examples[name].shape[1]))

    def getSubDataField(self, name, indices): 
        """
        Returns the data field according to the original un-permuted of data 
        """ 
        if (indices >= self.__numExamples).any() or (indices < 0).any(): 
            raise ValueError("Invalid example indices")
        
        if name not in self.__examples: 
            raise ValueError("Field does not exist: " + name)


        return self.__examples[name][ix_(indices)]

    def getSubExamplesList(self, indices):
        """
        Returns a new ExamplesList which contains these indices of the current one 
        """ 
        if (indices >= self.__numExamples).any() or (indices < 0).any(): 
            raise ValueError("Invalid example indices")
        
        numNewExamples = indices.shape[0]
        newExamplesList = ExamplesList(numNewExamples)
        
        for (name, value) in self.__examples.items(): 
            newExamplesList.addDataField(name, value[indices, :])

        newExamplesList.__labelsName = self.__labelsName
        newExamplesList.__defaultExamplesName = self.__defaultExamplesName
        
        return newExamplesList 
    
    def copyDataField(self, mle, name):  
        """
        Copy a data field into a new MLExample object 
        """ 
        if name not in self.__examples: 
            raise ValueError("Field does not exist: " + name)   
          
        mle.addDataField(name, self.getDataField(name))     
        return mle
    
    def randomSubData(self, number):
        """Set indices of the examples to a subset""" 
        if number < 0 or number > self.__numExamples: 
            raise ValueError("Random subset size must be between 0 and " + str(self.__numExamples))
     
        self.__exampleIndices = array(sample(list(range(0, self.__numExamples)), number))

    def permuteData(self):
        """ Permute the examples """ 
        self.__exampleIndices = permutation(self.__numExamples)
        
    def originalData(self):  
        """ Set the ordering of the examples back to the original """ 
        self.__exampleIndices = array(list(range(0, self.__numExamples)))  
    
    def getPermutationIndices(self):
        return self.__exampleIndices

    def getNumSampledExamples(self):
        return self.__exampleIndices.shape[0]

    def setPermutationIndices(self, indices):
        if (indices >= self.__numExamples).any() or (indices < 0).any(): 
            raise ValueError("Invalid example indices")
        
        self.__exampleIndices = indices 
    
    def getLabelsName(self):
        if self.__labelsName == 0: 
            raise ValueError("Name of labels not set")
        
        return self.__labelsName; 

    def setLabelsName(self, name):
        if name not in self.__examples: 
            raise ValueError("Data field name " + name + " does not exist.")
        
        self.__labelsName = name
        
    def getDefaultExamplesName(self):
        if self.__defaultExamplesName == 0: 
            raise ValueError("Name of default examples not set")
        
        return self.__defaultExamplesName; 

    def setDefaultExamplesName(self, name):
        if name not in self.__examples: 
            raise ValueError("Data field name " + name + " does not exist.")
        
        self.__defaultExamplesName = name

    def __dataFieldSizeToString(self, name):
        outputString = ""
        
        for dim in range(self.__examples[name].ndim): 
            outputString = outputString + str(self.__examples[name].shape[dim])
            if dim != self.__examples[name].ndim-1:
                outputString = outputString + "x"
            
        return outputString

    def __str__(self):
        outputString = "No. Examples: " + str(self.__numExamples) + "\n"
        for name in self.__examples:
            outputString = outputString + name + " size: " + self.__dataFieldSizeToString(name) + "\n"
        
        return  outputString

    def writeToMatFile(self, fileName):
        """
        Save only the data fields (and names) to a Matlab file. The .mat extension is 
        added to the file name. 
        """
        if self.__examples == {}: 
            raise ValueError("Trying to write an empty ExamplesList")
        
        appendMat = True
        format = '5'
        longFieldNames = False
        doCompression = True
        oneDAs = "column"
        
        io.savemat(fileName, self.__examples, appendMat, format, longFieldNames, doCompression, oneDAs)
        logging.info("Wrote ExamplesList to file " + fileName + ".mat")

    def getSampledExamplesLabels(self):
        """
        Returns the default examples and labels as numpy arrays.
        """
        
        exName = self.getDefaultExamplesName()
        labelsName = self.getLabelsName()
        X = self.getSampledDataField(exName)
        y = self.getSampledDataField(labelsName)
        
        if y.shape[1] == 1:
            y = y.ravel()

        return X, y

    def setSampledExamplesLabels(self, X, Y):
        """
        Sets the default examples and labels as numpy arrays. 
        """

        exName = "X"
        labelsName = "y"

        self.addDataField(exName, X)
        self.addDataField(labelsName, Y)
        self.setDefaultExamplesName(exName)
        self.setLabelsName(labelsName)

    @staticmethod
    def readFromMatFile(fileName):
        try: 
            examplesListDict = io.loadmat(fileName, struct_as_record=True)
        except IOError as error: 
            raise error
           
        del(examplesListDict["__version__"])
        del(examplesListDict["__header__"])
        del(examplesListDict["__globals__"])
        
        if len(examplesListDict) == 0: 
            raise ValueError("Bad ExamplesList file: contains no data fields")
        
        fieldNames = list(examplesListDict.keys())
        numExamples = examplesListDict[fieldNames[0]].shape[0]
        
        examplesList = ExamplesList(numExamples)
        
        for (name, value) in examplesListDict.items(): 
            examplesList.addDataField(name, value)
        
        logging.info("Read ExamplesList file " + fileName + " with " + str(numExamples) + " examples.")
        return examplesList 

    @staticmethod
    def readFromCsvFile(fileName):
        """
        Read a CSV file which stores a matrix. The last column must be the labels.
        """
        try:
             Xy = loadtxt(fileName, delimiter=",")
        except IOError as error:
            raise error

        numExamples = Xy.shape[0]
        numFeatures = Xy.shape[1]-1
        examplesList = ExamplesList(numExamples)
        
        examplesList.addDataField("X", Xy[:, 0:numFeatures])
        examplesList.addDataField("y", Xy[:, numFeatures])

        examplesList.setDefaultExamplesName("X")
        examplesList.setLabelsName("y")

        logging.info("Read ExamplesList file " + fileName + " with " + str(numExamples) + " examples.")

        return examplesList 

    @staticmethod
    def readFromFile(examplesFileName):
        if examplesFileName.find("mat") == len(examplesFileName)-3:
            examplesList = ExamplesList.readFromMatFile(examplesFileName)
        elif examplesFileName.find("csv") == len(examplesFileName)-3:
            examplesList = ExamplesList.readFromCsvFile(examplesFileName)
        else:
            raise ValueError("Invalid file name extension: " + + examplesFileName)

        return examplesList

    def __eq__(self, other):
        """ 
        Test if two classes are the same (by their data fields only)
        """
        if other.__numExamples != self.__numExamples: 
            return False 
        if list(other.__examples.keys()) != list(self.__examples.keys()): 
            return False 
        
        for (key, value) in self.__examples.items(): 
            if not (self.__examples[key] == other.__examples[key]).all(): 
                return False 
        
        return True 

    __numExamples = 0
    __exampleIndices = 0
    __labelsName = 0
    __defaultExamplesName = 0
    __examples = 0 