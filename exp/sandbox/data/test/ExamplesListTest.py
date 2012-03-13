'''
Created on 9 May 2009

@author: charanpal
'''
from apgl.data import ExamplesList
from apgl.util.PathDefaults import PathDefaults 
import unittest
import numpy
from numpy.random import rand

class ExamplesListTest(unittest.TestCase):

    """Set up some useful test variables"""
    def setUp(self):
        self.numExamples = 100
        self.numFeatures = 10
        self.fieldName = "X"
        
        self.ml = ExamplesList(self.numExamples)
        self.X = rand(self.numExamples, 10)
        self.ml.addDataField(self.fieldName, self.X)
        
    """ Test out the constructor """ 
    def testInit(self):
        ml = ExamplesList(100)
        
        self.assertEqual(ml.getNumExamples(),100)
        self.assertRaises(ValueError, ExamplesList, "abc") 
        self.assertRaises(ValueError, ExamplesList, -1) 
        
    def testAddExamples(self):
        ml = ExamplesList(100)
        X = rand(50, 10)
        Y = rand(100, 10)
        
        #Test adding wrong number of examples 
        self.assertRaises(ValueError, ml.addDataField, 'X', X)
        
        #Test adding duplicate fields
        ml.addDataField('Y', Y)
        self.assertRaises(ValueError, ml.addDataField, 'Y', Y)
        
        #Test the field has been added 
        self.assert_(numpy.array_equal(ml.getDataField('Y'), Y))
        
    def testDeleteExamples(self):
        #Test deleting a field that does not exist 
        self.assertRaises(ValueError, self.ml.deleteDataField, 'Y')

        #Test deleting the correct field
        self.ml.deleteDataField(self.fieldName)
        self.assertEqual(self.ml.getNumberOfDataFields(), 0)

    def testGetNumberOfDataFields(self):
        self.assertEqual(self.ml.getNumberOfDataFields(), 1)  
        
        Y = rand(self.numExamples, 20)
        self.ml.addDataField('Y', Y) 
        
        self.assertEqual(self.ml.getNumberOfDataFields(), 2)  
        
    def testDeleteAllExamples(self):
        Y = rand(self.numExamples, 20)
        self.ml.addDataField('Y', Y) 
        self.assertEqual(self.ml.getNumberOfDataFields(), 2) 
        
        self.ml.deleteAllDataFields()
        self.assertEqual(self.ml.getNumberOfDataFields(), 0) 

    def testGetDataFieldSize(self):
        self.assertEqual(self.ml.getDataFieldSize(self.fieldName, 0), self.numExamples) 
        self.assertEqual(self.ml.getDataFieldSize(self.fieldName, 1), self.numFeatures) 
        
        #Try to get the size of a field that does not exist
        self.assertRaises(ValueError, self.ml.getDataFieldSize, 'Y', 0)
        
        #Get the size of a dimensions that does not exist
        self.assertRaises(ValueError, self.ml.getDataFieldSize, self.fieldName, 2)
        self.assertRaises(ValueError, self.ml.getDataFieldSize, self.fieldName, -1)
        
        #Test out adding a vector and getting dimensionality 
        numExamples = 100
        y = rand(numExamples)
        
        ml = ExamplesList(numExamples)
        ml.addDataField('Y', y)
        
        self.assertEquals(ml.getDataFieldSize('Y', 0), numExamples)
        self.assertEquals(ml.getDataFieldSize('Y', 1), 1)

    def testGetDataFieldNames(self):
        numExamples = 100
        X = rand(numExamples, 10)
        Y = rand(numExamples, 20)
        
        ml = ExamplesList(numExamples)
        self.assertEqual(ml.getDataFieldNames(), list([])) 
        
        ml.addDataField('X', X)
        self.assertEqual(ml.getDataFieldNames(), list(['X'])) 
        
        ml.addDataField('Y', Y)
        self.assertEqual(ml.getDataFieldNames(), list(['Y', 'X'])) 
    
    def testGetDataField(self):
        self.assert_(numpy.array_equal(self.ml.getDataField(self.fieldName), self.X))
        self.assertRaises(ValueError, self.ml.getDataField, 'Y')
     

    def testGetSubDataField(self):
        numExamples = 100
        numFeatures = 10 
        
        ml = ExamplesList(numExamples)
        X = rand(numExamples, numFeatures)
        
        ml.addDataField("X", X)
        self.assertTrue((ml.getSubDataField("X", numpy.array([10, 12, 14])) == X[numpy.ix_([10,12,14]), :]).all())
        self.assertTrue((ml.getSubDataField("X", numpy.array([10, 12, 8, 2, 14])) == X[numpy.ix_([10,12,8, 2, 14]), :]).all())
    
        #Test error conditions 
        self.assertRaises(ValueError, ml.getSubDataField, "Y", numpy.array([10, 12, 14]))
        self.assertRaises(ValueError, ml.getSubDataField, "X", numpy.array([-1, 10, 12, 14]))
        self.assertRaises(ValueError, ml.getSubDataField, "X", numpy.array([0, 10, 12, numExamples]))

    def testPermuteData(self):
        self.ml.permuteData()
        X = self.ml.getDataField(self.fieldName)
        Xp = self.ml.getSampledDataField(self.fieldName)
        inds = self.ml.getPermutationIndices()
            
        self.assertEqual(Xp.shape[0], self.numExamples)
        
        #This might fail, but it's very unlikely 
        self.assertNotEqual(Xp, list(range(1, self.numExamples))) 
        self.assertTrue((X[numpy.ix_(inds), :] == Xp).all())
        
    def testGetRandomExamples(self):
        i = 5;     
        self.ml.randomSubData(i)  
        X = self.ml.getSampledDataField(self.fieldName)
        self.assertEqual(X.shape[0], i)
        
        self.ml.randomSubData(0)  
        X = self.ml.getSampledDataField(self.fieldName)
        self.assertEqual(X.shape[0], 0)
        
        self.assertRaises(ValueError, self.ml.getSampledDataField, 'Y')
        self.assertRaises(ValueError, self.ml.randomSubData, -1)
        self.assertRaises(ValueError, self.ml.randomSubData, self.numExamples+1)


    def testGetLabelsName(self):
        self.assertRaises(ValueError, self.ml.getLabelsName)
        self.assertRaises(ValueError, self.ml.setLabelsName, 'Y')
        self.ml.setLabelsName('X')
        
        self.assertEquals(self.ml.getLabelsName(), 'X')
        
    def testGetDefaultExamplesName(self):
        self.assertRaises(ValueError, self.ml.getDefaultExamplesName)
        self.assertRaises(ValueError, self.ml.setDefaultExamplesName, 'Y')
        self.ml.setDefaultExamplesName('X')
        
        self.assertEquals(self.ml.getDefaultExamplesName(), 'X')


    def testSetPermutationIndices(self):
        X = self.X
        
        #Test duplicate indices 
        indices = numpy.array([1, 1, 2, 10, 11])
        self.ml.setPermutationIndices(indices)
        X1 = self.ml.getSampledDataField(self.fieldName)
        self.assertTrue((X1 == X[indices, :]).all())
        
        #Test end points 
        indices = numpy.array([0, 1, 2, 15, self.numExamples-1])
        self.ml.setPermutationIndices(indices)
        X1 = self.ml.getSampledDataField(self.fieldName)
        self.assertTrue((X1 == X[indices, :]).all())
        
        #Check error conditions 
        indices = numpy.array([0, 5, -1])
        self.assertRaises(ValueError, self.ml.setPermutationIndices, indices)

        indices = numpy.array([0, self.numExamples])
        self.assertRaises(ValueError, self.ml.setPermutationIndices, indices)

        self.ml.originalData()
        
    def testOverwriteDataField(self):
        numExamples = 100
        ml = ExamplesList(numExamples)
        X = rand(numExamples, 10)
        X2 = rand(numExamples, 5)
        
        #Test overwriting a bad datafield
        self.assertRaises(ValueError, ml.overwriteDataField, 'X', X)
        
        ml.addDataField("X", X)
        self.assertTrue((ml.getDataField("X") == X).all())
        
        ml.overwriteDataField("X", X2)
        self.assertTrue((ml.getDataField("X") == X2).all())

    def testWriteToMatFile(self):
        numExamples = 100
        X = rand(numExamples, 10)
        ml = ExamplesList(numExamples)

        dir = PathDefaults.getTempDir()
        fileName = dir + "examplesList1"
        
        #Try writing with no examples 
        self.assertRaises(ValueError, ml.writeToMatFile, fileName)
        
        #Now, add a data field and try writing 
        ml.addDataField("X", X)
        ml.writeToMatFile(fileName)
        
        ml.addDataField("Y", X)
        ml.writeToMatFile(fileName)

        #Weird error: "The process cannot access the file because it is being used by another process"
        #os.remove(fileName + ".mat")
        
    def testReadFromMatFile(self):
        numExamples = 10
        dir = PathDefaults.getTempDir()
        fileName = dir + "examplesList1"
        X = rand(numExamples, 10)
        
        ml = ExamplesList(numExamples)
        ml.addDataField("X", X)
        ml.writeToMatFile(fileName)
        
        ml2 = ExamplesList.readFromMatFile(fileName)
        self.assertTrue(ml == ml2)

        Y = rand(numExamples, 20)

        ml.addDataField("Y", Y)
        ml.writeToMatFile(fileName)
        
        ml2 = ExamplesList.readFromMatFile(fileName)
        self.assertTrue(ml == ml2)
        
        Z = rand(numExamples, 50)

        ml.addDataField("Z", Z)
        ml.writeToMatFile(fileName)
        
        ml2 = ExamplesList.readFromMatFile(fileName)
        self.assertTrue(ml == ml2)

        #os.remove(fileName + ".mat")

    def testEq(self):
        numExamples = 10
        numXFeatures = 20 
        numYFeatures = 20 
        X = rand(numExamples, numXFeatures)
        Y = rand(numExamples, numYFeatures)
        
        ml = ExamplesList(numExamples)
        ml.addDataField("X", X)
        
        ml2 = ExamplesList(numExamples)
        ml2.addDataField("X", X)
        
        self.assertTrue(ml == ml2)
        
        ml2.addDataField("Y", Y)
        self.assertFalse(ml == ml2)
         
        ml.addDataField("Y", Y)
        self.assertTrue(ml == ml2)
        
        #Start again and test modified fields 
        X2 = X.copy()
        X2[0,0] = 50 
        ml = ExamplesList(numExamples)
        ml.addDataField("X", X)
        
        ml2 = ExamplesList(numExamples)
        ml2.addDataField("X", X2)
        
        self.assertFalse(ml == ml2)
        
        #Start again and test examples with different number of examples 
        ml = ExamplesList(numExamples)
        ml.addDataField("X", X)
        
        numExamples2 = numExamples * 2 
        X2 = rand(numExamples2, numXFeatures)
        
        ml2 = ExamplesList(numExamples2)
        ml2.addDataField("X", X2)
        
        self.assertFalse(ml == ml2)
        
    def testGetSubExamplesList(self):
        numExamples = 10
        numXFeatures = 20 
        numYFeatures = 20 
        X = rand(numExamples, numXFeatures)
        Y = rand(numExamples, numYFeatures)
        
        ml = ExamplesList(numExamples)
        ml.addDataField("X", X)
        ml.addDataField("Y", Y)
        ml.setDefaultExamplesName("X")
        ml.setLabelsName("Y")
        
        indices = numpy.array([5, 2, 7, 1])
        newExamplesList = ml.getSubExamplesList(indices)
        
        self.assertTrue((newExamplesList.getDataField("X") == ml.getDataField("X")[indices, :]).all())
        self.assertTrue((newExamplesList.getDataField("Y") == ml.getDataField("Y")[indices, :]).all())

        self.assertEquals(ml.getDefaultExamplesName(), newExamplesList.getDefaultExamplesName())
        self.assertEquals(ml.getLabelsName(), newExamplesList.getLabelsName())

    def testGetSampledExamplesLabels(self):
        numExamples = 10
        numXFeatures = 20
        numYFeatures = 20
        X = rand(numExamples, numXFeatures)
        Y = rand(numExamples, numYFeatures)

        ml = ExamplesList(numExamples)
        ml.addDataField("X", X)
        ml.addDataField("Y", Y)
        ml.setDefaultExamplesName("X")
        ml.setLabelsName("Y")

        (X2, Y2) = ml.getSampledExamplesLabels()

        self.assertTrue((X2 == X).all())
        self.assertTrue((Y2 == Y).all())

        #Test 1D labels
        y = rand(numExamples)

        ml = ExamplesList(numExamples)
        ml.addDataField("X", X)
        ml.addDataField("y", y)
        ml.setDefaultExamplesName("X")
        ml.setLabelsName("y")

        (X2, y2) = ml.getSampledExamplesLabels()

        self.assertTrue((X2 == X).all())
        self.assertTrue((y2 == y).all())

    def testReadFromCsvFile(self):
        dir = PathDefaults.getDataDir() + "test/"
        fileName = dir + "examplesList1.csv"

        examplesList = ExamplesList.readFromCsvFile(fileName)

        X = examplesList.getDataField(examplesList.getDefaultExamplesName())
        y = examplesList.getDataField(examplesList.getLabelsName())

        X2 = numpy.array([[10, 2], [4, -6], [24, 6]])
        y2 = numpy.array([[-1], [1], [-1]])

        self.assertTrue((X==X2).all())
        self.assertTrue((y==y2).all())

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(ExamplesListTest)
    unittest.TextTestRunner(verbosity=2).run(suite)


    