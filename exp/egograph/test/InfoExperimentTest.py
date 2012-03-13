

from apgl.egograph import * 
from apgl.graph import * 
from apgl.util import *
import unittest
import logging
import sys 


class InfoExperimentTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    def testSaveParams(self):
       outputDir = PathDefaults.getTempDir()
       paramsFileName = outputDir + "testParams"

       InfoExperiment.saveParams(paramsFileName)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()