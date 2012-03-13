'''
Created on 7 Aug 2009

@author: charanpal
'''
import logging
import sys
import unittest

from apgl.io.EgoCsvReader import EgoCsvReader
from apgl.util.PathDefaults import PathDefaults 
import numpy


class EgoCsvReaderTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    def tearDown(self):
        pass

    def testInit(self):
        eCsv = EgoCsvReader()

        self.assertEquals(len(eCsv.getEgoQuestionIds()), 62)
        self.assertEquals(len(eCsv.getAlterQuestionIds()), 62)

    def testReadFiles(self):
        p = 0.5
        eCsvReader = EgoCsvReader()
        eCsvReader.setP(p)

        dataDir = PathDefaults.getDataDir() + "infoDiffusion/"
        egoFileName = dataDir + "EgoData3.csv"
        alterFileName = dataDir + "AlterData10.csv"
        examplesList, egoIndicesR, alterIndices, egoIndicesNR, alterIndicesNR  = eCsvReader.readFiles(egoFileName, alterFileName)
        #logging.debug(examplesList.getDataField("X"))
        
        #Read in the ego and alter arrays 
        (egoArray, _) = eCsvReader.readFile(egoFileName, eCsvReader.getEgoQuestionIds())
        (alterArray, _) = eCsvReader.readFile(alterFileName, eCsvReader.getAlterQuestionIds())
        
        #Make up the correct results 
        numFeatures = examplesList.getDataFieldSize("X", 1)
        numPersonFeatures = numFeatures/2 

        #Note: no alters in this case 
        numTransmissons = 6
        X2 = numpy.zeros((numTransmissons, numFeatures))
        y2 = numpy.zeros((numTransmissons, 1))
        
        X2[0, 0:numPersonFeatures] = egoArray[0, :]
        X2[0, numPersonFeatures:numFeatures] = egoArray[1, :]
        y2[0, 0] = -1
        
        X2[1, 0:numPersonFeatures] = egoArray[0, :]
        X2[1, numPersonFeatures:numFeatures] = egoArray[2, :]
        y2[1, 0] = -1
        
        X2[2, 0:numPersonFeatures] = egoArray[1, :]
        X2[2, numPersonFeatures:numFeatures] = egoArray[0, :]
        y2[2, 0] = -1
        
        X2[3, 0:numPersonFeatures] = egoArray[1, :]
        X2[3, numPersonFeatures:numFeatures] = egoArray[2, :]
        y2[3, 0] = -1
        
        X2[4, 0:numPersonFeatures] = egoArray[2, :]
        X2[4, numPersonFeatures:numFeatures] = egoArray[0, :]
        y2[4, 0] = -1
        
        X2[5, 0:numPersonFeatures] = egoArray[2, :]
        X2[5, numPersonFeatures:numFeatures] = egoArray[1, :]
        y2[5, 0] = -1

        self.assertTrue((X2 == examplesList.getDataField("X")).all())
        self.assertTrue((y2 == examplesList.getDataField("y")).all())



        #Second test
        #================
        #I modified EgoData3 so that person 2 is the same age as person 1, and
        # hence a homophile of 1. She (2) is excluded from the non-receivers, since
        #she is a homophile of person 1.

        p = 0
        eCsvReader = EgoCsvReader()
        eCsvReader.setP(p)

        examplesList, egoIndicesR, alterIndices, egoIndicesNR, alterIndicesNR  = eCsvReader.readFiles(egoFileName, alterFileName)

        numTransmissons = 5
        X2 = numpy.zeros((numTransmissons, numFeatures))
        y2 = numpy.zeros((numTransmissons, 1))

        X2[0, 0:numPersonFeatures] = egoArray[0, :]
        X2[0, numPersonFeatures:numFeatures] = egoArray[2, :]
        y2[0, 0] = -1

        X2[1, 0:numPersonFeatures] = egoArray[1, :]
        X2[1, numPersonFeatures:numFeatures] = egoArray[0, :]
        y2[1, 0] = -1

        X2[2, 0:numPersonFeatures] = egoArray[1, :]
        X2[2, numPersonFeatures:numFeatures] = egoArray[2, :]
        y2[2, 0] = -1

        X2[3, 0:numPersonFeatures] = egoArray[2, :]
        X2[3, numPersonFeatures:numFeatures] = egoArray[0, :]
        y2[3, 0] = -1

        X2[4, 0:numPersonFeatures] = egoArray[2, :]
        X2[4, numPersonFeatures:numFeatures] = egoArray[1, :]
        y2[4, 0] = -1

        self.assertTrue((X2 == examplesList.getDataField("X")).all())
        self.assertTrue((y2 == examplesList.getDataField("y")).all())

    def testReplaceMissingValues(self):
        eCsvReader = EgoCsvReader()
        
        X = numpy.array([[1,2], [5,6], [0, 3]])
        Xnew = eCsvReader.replaceMissingValues(X)

        self.assertTrue((Xnew == numpy.array([[1,2], [5,6], [3, 3]])).all())

    def testReplaceMissingValues2(self):
        eCsvReader = EgoCsvReader()

        numpy.random.seed(0)
        X = numpy.array([[1,2], [5,6], [0, 3]])
        Xnew = eCsvReader.replaceMissingValues2(X)
        
        self.assertTrue((Xnew == numpy.array([[1,2], [5,6], [10, 3]])).all())
        
    def testCsvRowToVector(self):
        eCsvReader = EgoCsvReader()
        
        csvRow = ["1", "5", "2", "12", ""]
        csvRow2 = ["2", "4", "8", "2", "1"]
        csvTitles = ["A", "B", "C", "D", "E"]
        csvErrorTitles1 = ["A", "B", "C", "D", "E" ,"F"]
        csvErrorTitles2 = ["A", "B", "C"]
        
        questionIds = [("B", 0), ("A", 0), ("E", 1)]
        questionIdsError1 = [("B", 0), ("A", 0), ("E", 0)]
        questionIdsError2 = [("B", 0), ("A", 0), ("Z", 1)]
        
        self.assertRaises(ValueError, eCsvReader.csvRowToVector, csvRow, questionIds, csvErrorTitles1)
        self.assertRaises(ValueError, eCsvReader.csvRowToVector, csvRow, questionIds, csvErrorTitles2)
        
        v = eCsvReader.csvRowToVector(csvRow, questionIds, csvTitles)
        self.assertTrue((v==numpy.array([5, 1, 0])).all())
        
        v = eCsvReader.csvRowToVector(csvRow2, questionIds, csvTitles)
        self.assertTrue((v==numpy.array([4, 2, 1])).all())
                        
        self.assertRaises(ValueError, eCsvReader.csvRowToVector, csvRow, questionIdsError1, csvTitles)
        self.assertRaises(ValueError, eCsvReader.csvRowToVector, csvRow, questionIdsError2, csvTitles)

    def testReadFile(self): 
        eCsvReader = EgoCsvReader()
        #logging.debug(os.getcwd())
        dir = PathDefaults.getDataDir()
        fileName = dir + "test/TestData.csv"
        questionIds = [("Q14", 0), ("Q12", 1) , ("Q2", 0)]

        missing = 1
        (X, titles) = eCsvReader.readFile(fileName, questionIds, missing)
        
        X2 = numpy.zeros((10, 3))
        X2[0, :] = [0.621903386,0.608560354,0.33290608]
        X2[1, :] = [0.318548924,0.402390713,0.129956291]
        X2[2, :] = [0.956658404,0.344317772,0.680386616]
        X2[3, :] = [0.267607668,0.119647983,0.116893619]
        X2[4, :] = [0.686589498,0.402390713,0.426789174]
        X2[5, :] = [0.373575769,0.025846789,0.797125005]
        X2[6, :] = [0.493793948,0.402390713,0.990507109]
        X2[7, :] = [0.524534585,0.525169385,0.772917183]
        X2[8, :] = [0.339055395,0.402390713,0.684788001]
        X2[9, :] = [0.997774183,0.790801992,0.643252009]
        
        self.assertAlmostEquals(numpy.linalg.norm(X-X2),0, places=6)
         
    def testGenerateReceivers(self): 
        eCsvReader = EgoCsvReader()
        
        numAlters = 10
        numPartialFields = 11
        alterFields = numPartialFields + 2
        alterFieldIndices = list(range(0, numPartialFields))
        
        egoAlterArray = numpy.zeros((2, 45))
        alterArray = numpy.zeros((numAlters, alterFields))
        
        egoAlterArray[0, 0:11] = [2, 12, 1, 0, 0, 0, 0, 0, 0, 0, 1]
        egoAlterArray[0, 11:22] = [1, 11, 1, 0, 0, 0, 0, 0, 0, 0, 1]
        egoAlterArray[1, 0:11] = [5, 6, 1, 0, 0, 0, 0, 0, 0, 0, 3]
        
        alterArray[0, :] = [2, 12, 1, 0, 0, 0, 0, 0, 0, 0, 1, 12, 13]
        alterArray[1, :] = [1, 11, 1, 0, 0, 0, 0, 0, 0, 0, 1, 84, 12]
        alterArray[2, :] = [2, 12, 1, 0, 0, 0, 1, 0, 0, 0, 0, 4, 9]
        alterArray[4, :] = [5, 6, 1, 0, 0, 0, 0, 0, 0, 0, 3, 34, 12]
        
        generatedAltersArray2 = numpy.zeros((3, alterFields))
        generatedAltersArray2[0, :] = alterArray[0, :]
        generatedAltersArray2[1, :] = alterArray[1, :]
        generatedAltersArray2[2, :] = alterArray[4, :]
        
        (generatedAltersArray, egoIndices, alterIndices) = eCsvReader.generateReceivers(egoAlterArray, alterArray, alterFieldIndices)

        self.assertTrue((egoIndices == numpy.array([0,0,1])).all())
        self.assertTrue((alterIndices == numpy.array([0,1,4])).all())
        self.assertTrue((generatedAltersArray == generatedAltersArray2).all())
        
        #2nd test 
        egoAlterArray[1, 11:22] = [2, 12, 1, 0, 0, 0, 1, 0, 0, 0, 0]
        
        (generatedAltersArray, egoIndices, alterIndices) = eCsvReader.generateReceivers(egoAlterArray, alterArray, alterFieldIndices)
        
        generatedAltersArray2 = numpy.zeros((4, alterFields))
        generatedAltersArray2[0, :] = alterArray[0, :]
        generatedAltersArray2[1, :] = alterArray[1, :]
        generatedAltersArray2[2, :] = alterArray[4, :]
        generatedAltersArray2[3, :] = alterArray[2, :]
        self.assertTrue((egoIndices == numpy.array([0,0,1,1])).all())
        self.assertTrue((alterIndices == numpy.array([0,1,4,2])).all())
        self.assertTrue((generatedAltersArray == generatedAltersArray2).all())
        
    def testGenerateNonReceivers(self):
        numEgos = 3
        numFeatures = 5

        p = 1
        eCsvReader = EgoCsvReader()
        eCsvReader.setP(p)

        numContactsIndices = [0, 1]
        homophileIndexPairs = [(2,3)]

        receiverCounts = numpy.zeros(numEgos)

        #First test a very simple example with 1 homophile pair
        egoArray = numpy.zeros((numEgos, numFeatures))
        egoArray[0, :] = [0, 1, 1, 5, 4]
        egoArray[1, :] = [0, 1, 1, 5, 8]
        egoArray[2, :] = [0, 0, 1, 3, 6]

        (contactsArray, egoIndices, alterIndices) = eCsvReader.generateNonReceivers(egoArray, numContactsIndices, homophileIndexPairs, receiverCounts)

        numContacts = 4
        contactsArray2 = numpy.zeros((numContacts, numFeatures))
        contactsArray2[0, :] = [0, 1, 1, 5, 8]
        contactsArray2[1, :] = [0, 0, 1, 3, 6]
        contactsArray2[2, :] = [0, 1, 1, 5, 4]
        contactsArray2[3, :] = [0, 0, 1, 3, 6]


        egoIndices2 = numpy.zeros(numContacts)
        egoIndices2[0] = 0
        egoIndices2[1] = 0
        egoIndices2[2] = 1
        egoIndices2[3] = 1

        self.assertTrue((contactsArray == contactsArray2).all())
        self.assertTrue((egoIndices == egoIndices2).all())

        #Test the case when there are some receivers already
        receiverCounts = numpy.array([1,1,1])
        (contactsArray, egoIndices, alterIndices) = eCsvReader.generateNonReceivers(egoArray, numContactsIndices, homophileIndexPairs, receiverCounts)


        numContacts = 2
        contactsArray2 = numpy.zeros((numContacts, numFeatures))
        contactsArray2[0, :] = [0, 1, 1, 5, 8]
        contactsArray2[1, :] = [0, 1, 1, 5, 4]


        egoIndices2 = numpy.zeros(numContacts)
        egoIndices2[0] = 0
        egoIndices2[1] = 1

        self.assertTrue((contactsArray == contactsArray2).all())
        self.assertTrue((egoIndices == egoIndices2).all())

        #A more complex example
        numEgos = 6
        egoArray = numpy.zeros((numEgos, numFeatures))
        egoArray[0, :] = [1, 1, 1, 5, 4]
        egoArray[1, :] = [0, 0, 1, 5, 8]
        egoArray[2, :] = [0, 0, 1, 3, 6]
        egoArray[3, :] = [0, 0, 1, 5, 1]
        egoArray[4, :] = [0, 0, 1, 5, 2]
        egoArray[5, :] = [0, 0, 1, 5, 3]

        receiverCounts = numpy.zeros(numEgos)
        (contactsArray, egoIndices, alterIndices) = eCsvReader.generateNonReceivers(egoArray, numContactsIndices, homophileIndexPairs, receiverCounts)

        numContacts = 4
        contactsArray2 = numpy.zeros((numContacts, numFeatures))
        contactsArray2[0, :] = [0, 0, 1, 5, 8]
        contactsArray2[1, :] = [0, 0, 1, 5, 1]
        contactsArray2[2, :] = [0, 0, 1, 5, 2]
        contactsArray2[3, :] = [0, 0, 1, 5, 3]

        egoIndices2 = numpy.zeros(numContacts)
        egoIndices2[0] = 0
        egoIndices2[1] = 0
        egoIndices2[2] = 0
        egoIndices2[3] = 0


        self.assertTrue((contactsArray == contactsArray2).all())
        self.assertTrue((egoIndices == egoIndices2).all())

        #Test picking non-homophiles
        egoArray[0, :] = [2, 1, 2, 5, 4]

        (contactsArray, egoIndices, alterIndices) = eCsvReader.generateNonReceivers(egoArray, numContactsIndices, homophileIndexPairs, receiverCounts)

        numContacts = 5
        contactsArray2 = numpy.zeros((numContacts, numFeatures))
        contactsArray2[0, :] = [0, 0, 1, 5, 8]
        contactsArray2[1, :] = [0, 0, 1, 3, 6]
        contactsArray2[2, :] = [0, 0, 1, 5, 1]
        contactsArray2[3, :] = [0, 0, 1, 5, 2]
        contactsArray2[4, :] = [0, 0, 1, 5, 3]

        egoIndices2 = numpy.zeros(numContacts)
        egoIndices2[0] = 0
        egoIndices2[1] = 0
        egoIndices2[2] = 0
        egoIndices2[3] = 0
        egoIndices2[4] = 0


        self.assertTrue((contactsArray == contactsArray2).all())
        self.assertTrue((egoIndices == egoIndices2).all())

        #Choose different p 
        p = 0.5
        eCsvReader.setP(p)

        egoArray[0, :] = [1, 1, 1, 5, 4]
        egoArray[1, :] = [0, 0, 1, 5, 8]
        egoArray[2, :] = [0, 0, 1, 3, 6]
        egoArray[3, :] = [0, 0, 1, 5, 8]
        egoArray[4, :] = [0, 0, 1, 5, 8]
        egoArray[5, :] = [0, 0, 1, 5, 8]

        (contactsArray, egoIndices, alterIndices) = eCsvReader.generateNonReceivers(egoArray, numContactsIndices, homophileIndexPairs, receiverCounts)

        numContacts = 3
        contactsArray2 = numpy.zeros((numContacts, numFeatures))
        contactsArray2[0, :] = [0, 0, 1, 5, 8]
        contactsArray2[1, :] = [0, 0, 1, 5, 8]
        contactsArray2[2, :] = [0, 0, 1, 3, 6]


        egoIndices2 = numpy.zeros(numContacts)
        egoIndices2[0] = 0
        egoIndices2[1] = 0
        egoIndices2[2] = 0

        self.assertTrue((contactsArray == contactsArray2).all())
        self.assertTrue((egoIndices == egoIndices2).all())

        #Test 2 different homophile fields
        p = 1
        eCsvReader.setP(p)
        numEgos = 6
        numFeatures = 7

        homophileIndexPairs = [(2,3), (4,5)]

        egoArray = numpy.zeros((numEgos, numFeatures))
        egoArray[0, :] = [0, 0, 1, 5, 1, 2, 1]
        egoArray[1, :] = [0, 0, 1, 5, 1, 3, 2]
        egoArray[2, :] = [0, 0, 1, 4, 1, 2, 3]
        egoArray[3, :] = [1, 0, 1, 5, 1, 2, 4]
        egoArray[4, :] = [0, 0, 1, 2, 1, 1, 5]
        egoArray[5, :] = [0, 0, 1, 5, 1, 2, 6]

        (contactsArray, egoIndices, alterIndices) = eCsvReader.generateNonReceivers(egoArray, numContactsIndices, homophileIndexPairs, receiverCounts)

        numContacts = 2
        contactsArray2 = numpy.zeros((numContacts, numFeatures))
        contactsArray2[0, :] = [0, 0, 1, 5, 1, 2, 1]
        contactsArray2[1, :] = [0, 0, 1, 5, 1, 2, 6]

        egoIndices2 = numpy.zeros(numContacts)
        egoIndices2[0] = 3
        egoIndices2[1] = 3

        self.assertTrue((contactsArray == contactsArray2).all())
        self.assertTrue((egoIndices == egoIndices2).all())

        

    def testEgoAlterDistributions(self):
        """
        Check that the distributions of the egos and alters are about the same 
        """

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()