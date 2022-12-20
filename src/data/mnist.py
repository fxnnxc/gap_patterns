

from .base_data import BaseData

class MNIST(BaseData):
    def __init__(self, flags):
        pass 
    
    def __getitem__(self, x):
        x = self.common_preprocess(x) 
        return x
    
    
    