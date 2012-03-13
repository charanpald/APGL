'''
Created on 10 Jul 2009

@author: charanpal

Test out EgoProbabilities. 
'''

import unittest


class EgoProbabilitiesTest(unittest.TestCase):
    def setUp(self):
        pass
        
    def tearDown(self):
        pass

    def testInit(self):
        pass        
    

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(EgoProbabilitiesTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
    