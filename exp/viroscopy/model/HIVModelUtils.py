

"""
Keep some default parameters for the epidemic model. 
"""

class HIVModelUtils(object):
    def __init__(self): 
        pass 
    
    @staticmethod
    def defaultTheta(): 
        theta = [50, 1.0, 0.5, 1.0/800, 0.01, 0.05, 0.1, 38.0/1000, 30.0/1000, 170.0/1000]
        return theta 
        
    @staticmethod 
    def defaultSimulationParams(): 
        T = 1000.0
        recordStep = 90
        printStep = 10
        M = 2000
        
        return T, recordStep, printStep, M 
        
    