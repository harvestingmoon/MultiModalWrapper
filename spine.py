from layers import Base,GPU_Base
from loaders import Batch,Input_Loader,Adaptor
from helper import get_key_from_value

class Pipeline:
    def __init__(self,*layers):
        for l in layers: 
            if not isinstance(l,GPU_Base) or not isinstance(l,Base):
                return "Layer needs to be either of class GPU_Base or Base!"
        self.dct = {}
        temp = None
        for l in self.layers:
            if temp:
                temp = Adaptor(temp,l)
                self.lst[temp.name] = temp
            temp = l

    # Takes in input and input is a class that contains both input_loader and model itself
    def forward_batch(self,inpt,train = False,override = False):
        if not isinstance(inpt,Batch):
            return "Invalid Input!"
        
        if len(inpt.lst) == 2: 
            a = Adaptor(inpt.lst[0],inpt.lst[1])
            return a.fwd(inpt,train)
        else: 
            lst_keys = list(self.dct.values())
            l1_output = lst_keys[0].fwd(inpt,train = train,override = override)
            temp = l1_output
            for l in lst_keys[1:]:
                temp = l.select_pass(temp,train = train,override = override)
            
            return temp
        
        
    ''' Selective Layer Forwarding!'''
    def layer_fwd(self,layer,inpt,train = False, override = False,layer_name = None):
        if layer_name in self.dct or layer in self.dct.values():
            if layer_name:
                x = self.dct[layer_name]
                return x.fwd(inpt,train = train,override = override,same = False)
            else:
                return layer.fwd(inpt,train = train,override = override,same = False)
        else:
            return f"{layer} is not found in spine, check the name again!"
        
            


        
            