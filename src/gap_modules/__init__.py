from .base_model import BaseModel
from .simple_gap import SimpleGAP

def get_gap_module(flags):
    name = flags.model_name
    if name== "test":
        return BaseModel(flags)
    elif name =="simple":
        return SimpleGAP(flags)
    
