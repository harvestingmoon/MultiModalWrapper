class Input_Loader:
    def __init__(self,model,input):
        self.model = model
        self.input = input

class Batch:
    def __init__(self,loaders): # list of input_loaders 
        self.x = {}
        self.is_gpu = False
        for l in loaders:
            if not isinstance(l,Input_Loader):
                raise Exception
            else:
                self.x[l.model] = l.input