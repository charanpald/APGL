from apgl.util.Util import Util 

class AbstractPreprocessor(object):
    '''
    An abstract preprocessor object for a set of examples. 
    '''

    def __init__(self):
        '''
        Constructor
        '''
        Util.abstract()

    def learn(self, X):
        Util.abstract()

    def process(self, X):
        Util.abstract()