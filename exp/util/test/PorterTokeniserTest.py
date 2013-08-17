import unittest
from exp.util.PorterTokeniser import PorterTokeniser

class  PorterTokeniserTest(unittest.TestCase):
    def setUp(self): 
        pass 
        
    def testCall(self): 
        tokeniser = PorterTokeniser() 
        
        doc = "System and human-system engineering testing of EPS."
        
        print(tokeniser(doc))
        
        
if __name__ == '__main__':
    unittest.main()
