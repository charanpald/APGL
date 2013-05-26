
from exp.sandbox.recommendation.SGDNorm2Reg import SGDNorm2Reg

"""
An iterative version of matrix factoisation using frobenius norm penalisation. 
"""

class IterativeSGDNorm2Reg(object): 
    def __init__(self, k, lmbda, eps, tmax): 
        self.baseLearner = SGDNorm2Reg(k, lmbda, eps, tmax)
        
    def learnModel(self, XIterator): 
        
        class ZIterator(object):
            def __init__(self, XIterator, baseLearner):
                self.XIterator = XIterator 
                self.baseLearner = baseLearner 

            def __iter__(self):
                return self

            def next(self):
                X = self.XIterator.next()
                
                #Return the matrices P, Q as the learnt model 
                return self.baseLearner.learnModel(X, storeAll=False)
                
        return ZIterator(XIterator, self.baseLearner)
        

    def predict(self, ZIter, indList):
        """
        Make a set of predictions for a given iterator of completed matrices and
        an index list.
        """
        class ZTestIter(object):
            def __init__(self, baseLearner):
                self.i = 0
                self.baseLearner = baseLearner

            def __iter__(self):
                return self

            def next(self):
                Z = next(ZIter) 
                Xhat = self.baseLearner.predict(Z, indList[self.i])  
                self.i += 1
                return Xhat 

        return ZTestIter(self.baseLearner)