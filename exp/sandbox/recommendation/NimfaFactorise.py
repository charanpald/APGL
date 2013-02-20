"""
A wrapper for the matrix factorisation in Nimfa. 
"""

class NimfaFactorise(object): 
    
    def __init__(self, method, rank): 
        self.method = method  
        self.rank = rank 
    
    
    def learnModel(self, X): 
        
        