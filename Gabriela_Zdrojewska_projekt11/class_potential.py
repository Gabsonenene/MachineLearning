import numpy as np 

"""
Klasa potential 
"""

class potential:
    def __init__(self, variables = np.array([]), table = np.array([])):
        self.variables = variables 
        self.table = table 

