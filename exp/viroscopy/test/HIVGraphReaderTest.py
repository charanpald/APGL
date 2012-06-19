
import apgl
import numpy 
import unittest
import pickle 
import numpy.testing as nptst 

from exp.viroscopy.HIVGraphReader import HIVGraphReader

class  HIVGraphReaderTest(unittest.TestCase):
    def setup(self):
        pass
    
    def testreadSimulationHIVGraph(self): 
        
        hivReader = HIVGraphReader()
        graph = hivReader.readSimulationHIVGraph()
        
        
        print(graph)
        #TODO: Test this in much more detail 

if __name__ == '__main__':
    unittest.main()

