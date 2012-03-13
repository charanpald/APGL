import numpy 
import unittest
import logging
import pywt
from apgl.metabolomics.MetabolomicsUtils import MetabolomicsUtils 

class  MetabolomicsUtilsTestCase(unittest.TestCase):
    def testCreateIndicatorLabels(self):
        numpy.set_printoptions(threshold=3000)
        X, X2, Xs, Xopls, YList, df = MetabolomicsUtils.loadData()

        #YList = MetabolomicsUtils.createLabelList(df, MetabolomicsUtils.getLabelNames())

        Y1, inds1 = YList[0]
        Y2, inds2 = YList[1]
        Y3, inds3 = YList[2]

        YIgf1Inds, YICortisolInds, YTestoInds = MetabolomicsUtils.createIndicatorLabels(YList)

        s = YIgf1Inds[0] + YIgf1Inds[1] + YIgf1Inds[2]
        self.assertTrue((s == numpy.ones(s.shape[0])).all())

        s = YICortisolInds[0] + YICortisolInds[1] + YICortisolInds[2]
        self.assertTrue((s == numpy.ones(s.shape[0])).all())

        s = YTestoInds[0] + YTestoInds[1] + YTestoInds[2]
        self.assertTrue((s == numpy.ones(s.shape[0])).all())

        #Now compare to those labels in the file
        labelNames = ["Ind.Testo.1", "Ind.Testo.2", "Ind.Testo.3"]
        labelNames.extend(["Ind.Cortisol.1", "Ind.Cortisol.2", "Ind.Cortisol.3"])
        labelNames.extend(["Ind.IGF1.1", "Ind.IGF1.2", "Ind.IGF1.3"])

        Y = numpy.array(df.rx(labelNames[6])).ravel()[inds1]
        logging.debug(numpy.sum(numpy.abs(YIgf1Inds[0] - Y)))
        Y = numpy.array(df.rx(labelNames[7])).ravel()[inds1]
        logging.debug(numpy.sum(numpy.abs(YIgf1Inds[1] - Y)))
        Y = numpy.array(df.rx(labelNames[8])).ravel()[inds1]
        logging.debug(numpy.sum(numpy.abs(YIgf1Inds[2] - Y)))

        Y = numpy.array(df.rx(labelNames[3])).ravel()[inds2]
        logging.debug(numpy.sum(numpy.abs(YICortisolInds[0] - Y)))
        Y = numpy.array(df.rx(labelNames[4])).ravel()[inds2]
        logging.debug(numpy.sum(numpy.abs(YICortisolInds[1] - Y)))
        Y = numpy.array(df.rx(labelNames[5])).ravel()[inds2]
        logging.debug(numpy.sum(numpy.abs(YICortisolInds[2] - Y)))

        Y = numpy.array(df.rx(labelNames[0])).ravel()[inds3]
        logging.debug(numpy.sum(numpy.abs(YTestoInds[0] - Y)))
        Y = numpy.array(df.rx(labelNames[1])).ravel()[inds3]
        logging.debug(numpy.sum(numpy.abs(YTestoInds[1] - Y)))
        Y = numpy.array(df.rx(labelNames[2])).ravel()[inds3]
        logging.debug(numpy.sum(numpy.abs(YTestoInds[2] - Y)))


    def testGetWaveletFeaturesTest(self):
        #See if we can reproduce the data from the wavelet 

        X, X2, Xs, Xopls, YList, df = MetabolomicsUtils.loadData()

        waveletStr = 'db4'
        mode = "zpd"
        level = 10
        C = pywt.wavedec(X[0, :], waveletStr, level=level, mode=mode)
        X0 = pywt.waverec(C, waveletStr, mode)
        tol = 10**-6
        self.assertTrue(numpy.linalg.norm(X0 - X[0, :]) < tol)

        def reconstructSignal(X, Xw, waveletStr, level, mode, C):
            Xrecstr = numpy.zeros(X.shape)
            
            for i in range(Xw.shape[0]):
                C2 = []

                colIndex = 0
                for j in range(len(list(C))):
                    C2.append(Xw[i, colIndex:colIndex+len(C[j])])
                    colIndex += len(C[j])

                Xrecstr[i, :] = pywt.waverec(tuple(C2), waveletStr, mode)

            return Xrecstr

        #Now do the same for the whole X
        C = pywt.wavedec(X[0, :], waveletStr, level=level, mode=mode)
        Xw = MetabolomicsUtils.getWaveletFeatures(X, waveletStr, level, mode)
        Xrecstr = reconstructSignal(X, Xw, waveletStr, level, mode, C)
        self.assertTrue(numpy.linalg.norm(X - Xrecstr) < tol)

        waveletStr = 'db8'
        C = pywt.wavedec(X[0, :], waveletStr, level=level, mode=mode)
        Xw = MetabolomicsUtils.getWaveletFeatures(X, waveletStr, level, mode)
        Xrecstr = reconstructSignal(X, Xw, waveletStr, level, mode, C)
        self.assertTrue(numpy.linalg.norm(X - Xrecstr) < tol)

        waveletStr = 'haar'
        C = pywt.wavedec(X[0, :], waveletStr, level=level, mode=mode)
        Xw = MetabolomicsUtils.getWaveletFeatures(X, waveletStr, level, mode)
        Xrecstr = reconstructSignal(X, Xw, waveletStr, level, mode, C)
        self.assertTrue(numpy.linalg.norm(X - Xrecstr) < tol)
        
    def testScoreLabel(self):#
        numExamples = 10 
        Y = numpy.random.rand(numExamples)

        bounds = numpy.array([0, 0.2, 0.8, 1.0])

        YScores = MetabolomicsUtils.scoreLabels(Y, bounds)

        inds1 = numpy.argsort(Y)
        inds2 = numpy.argsort(YScores[:, 0])
        inds3 = numpy.argsort(YScores[:, -1])

        inds4 = numpy.argsort(numpy.abs(Y - 0.5))
        inds5 = numpy.argsort(YScores[:, 1])

        self.assertTrue((inds1 == inds3).all())
        self.assertTrue((inds1 == numpy.flipud(inds2)).all())
        self.assertTrue((inds4 == numpy.flipud(inds5)).all())

        #Test we don't get problems when Y has the same values
        Y = numpy.ones(numExamples)
        YScores = MetabolomicsUtils.scoreLabels(Y, bounds)

        self.assertTrue((YScores == numpy.ones((Y.shape[0], 3))).all())

    def testReconstructSignal(self):
        numExamples = 100 
        numFeatures = 16 
        X = numpy.random.rand(numExamples, numFeatures)

        level = 10 
        mode = "cpd"
        waveletStr = "db4"
        C = pywt.wavedec(X[0, :], waveletStr, mode, level=10)

        Xw = MetabolomicsUtils.getWaveletFeatures(X, waveletStr, level, mode)
        X2 = MetabolomicsUtils.reconstructSignal(X, Xw, waveletStr, mode, C)

        tol = 10**-6 
        self.assertTrue(numpy.linalg.norm(X - X2) < tol)

    def testFilterWavelet(self):
        numExamples = 100
        numFeatures = 16
        X = numpy.random.rand(numExamples, numFeatures)

        level = 10
        mode = "cpd"
        waveletStr = "db4"
        C = pywt.wavedec(X[0, :], waveletStr, mode, level=10)

        Xw = MetabolomicsUtils.getWaveletFeatures(X, waveletStr, level, mode)
        
        N = 10
        Xw2, inds = MetabolomicsUtils.filterWavelet(Xw, N)

        tol = 10**-6 
        self.assertEquals(inds.shape[0], N)
        self.assertTrue(numpy.linalg.norm( Xw[:, inds] - Xw2[:, inds] ) < tol)

        zeroInds = numpy.setdiff1d(numpy.arange(Xw.shape[1]), inds)
        self.assertTrue(numpy.linalg.norm(Xw2[:, zeroInds]) < tol)




        

if __name__ == '__main__':
    unittest.main()

