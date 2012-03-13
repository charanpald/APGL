class VqUtils(object):
    @staticmethod 
    def whiten(obs):
        """
          care about null standard deviation
        """
        mean = obs.mean(axis=0)
        stdDev = obs.std(axis=0)
        stdDev[stdDev==0] = 1
        return (obs-mean)/stdDev
        
    


