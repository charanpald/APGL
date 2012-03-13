import numpy
import unittest
from apgl.util.PathDefaults import PathDefaults

class  GenerateToyDataTest(unittest.TestCase):

    def testToyData(self):
        dataDir = PathDefaults.getDataDir() + "modelPenalisation/toy/"
        data = numpy.load(dataDir + "toyData.npz")
        gridPoints, X, y, pdfX, pdfY1X, pdfYminus1X = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"], data["arr_4"], data["arr_5"]


        pxSum = 0
        pY1XSum = 0
        pYminus1XSum = 0

        px2Sum = 0 
        squareArea = (gridPoints[1]-gridPoints[0])**2

        for i in range(gridPoints.shape[0]-1):
            for j in range(gridPoints.shape[0]-1):
                px = (pdfX[i,j]+pdfX[i+1,j]+pdfX[i, j+1]+pdfX[i+1, j+1])/4
                pxSum += px*squareArea

                pY1X = (pdfY1X[i,j]+pdfY1X[i+1,j]+pdfY1X[i, j+1]+pdfY1X[i+1, j+1])/4
                pY1XSum += pY1X*squareArea

                pYminus1X = (pdfYminus1X[i,j]+pdfYminus1X[i+1,j]+pdfYminus1X[i, j+1]+pdfYminus1X[i+1, j+1])/4
                pYminus1XSum += pYminus1X*squareArea

                px2Sum += px*pY1X*squareArea + px*pYminus1X*squareArea

        self.assertAlmostEquals(pxSum, 1)
        print(pY1XSum)
        print(pYminus1XSum)

        self.assertAlmostEquals(px2Sum, 1)

if __name__ == '__main__':
    unittest.main()

