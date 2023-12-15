from layers import Base,GPU_Base
from loaders import Batch,Input_Loader
class Pipeline:
    def __init__(self,*layers):
        for l in layers: 
            if not isinstance(l,GPU_Base) or not isinstance(l,Base):
                return "Layer needs to be either of class GPU_Base or Base!"
            
        self.layers = {l:True for l in layers}

    # Takes in input and input is a class that contains both input_loader and model itself
    def forward_batch(self,inpt,train = False):
        if not isinstance(inpt,Batch):
            return "Invalid Input!"
        # First layer
        l1 = self.layers[0]
        if len(inpt) == 1:
            if train:
                output = l1.train_same_input(inpt)
            else:
                output = l1.eval_same_input(inpt)
        else:
            if train:
                output = l1.train_diff_input(inpt)
            else: 
                output = l1.eval_diff_input(inpt)
        
        
        
        
        
            