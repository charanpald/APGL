
import time
from apgl.util.PathDefaults import PathDefaults 



while True: 
    filename = PathDefaults.getOutputDir() + "deleteMe"
    file = open(filename, "w")
    
    file.write("This is a test")
    file.close() 
    print("Wrote to file " + filename)
    
    time.sleep(10)
    
    

