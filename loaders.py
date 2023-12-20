from torch import Tensor
class Input_Loader:
    def __init__(self,model,input):
        self.model = model
        self.input = input

class Batch:
    def __init__(self,loaders): # list of input_loaders 
        self.x = {}
        self.is_gpu = False
        self.lst = loaders
        for l in loaders:
            if not isinstance(l,Input_Loader):
                raise Exception
            else:
                self.x[l.model] = l.input
    def vals(self):
        return self.x.values()
class Adaptor:
    ''' Dual adaptor to connect top and bottom layers together, this is required 
    for build up of actual spine. 
    Note that in this context: Model is referred to as a chunk of the Base Layer '''
    def __init__(self,m1,m2) -> None:
        self.top = m1
        self.bottom = m2
        self.name = f"{self.top.name()} => {self.bottom.name()}"

    def indv_connection(self):
        self.connect_all()
        return f"{self.top.name()} => {self.bottom.name()} done!"
    
    def connect_all(self):
        if len(self.top) != len(self.bottom):
            return f"Length of {self.top.name()} of size {len(self.top)} != length of {self.bottom.layer_name} of size {len(self.bottom)}, check the model again!"
        else:
            m1 = self.top
            m2 = self.bottom
            for model in m1.models:
                for _model in m2.models:
                    m1.indv_connect(model,_model)
                    m2.indv_connect(_model,model)
        return f"{m1} is fully connected to {m2}!"
    def manual_connect(self,model1,model2):
        if model1 not in self.top.get_models():
            return f"Model {model1} is not found in {self.top.name()}!"
        if model2 not in self.bottom.get_models():
            return f"Model {model2} is not found in {self.bottom.name()}!"
        m1 = self.top
        m2 = self.bottom
        m1.indv_connect(model1,model2)
        m2.indv_connect(model2,model1)
        return f"Model {model1} from {m1.name()} is connected to Model {model2} from {m2.name()}"
    
    def fwd(self,input_,train = False,size = True,override = False,same = False):
        inpt_tensor = []
        if isinstance(input_,Tensor): 
            for model in self.models:
                inpt_tensor.append(Input_Loader(model,input_))
            input_ = Batch(inpt_tensor)

        for inpt_loader in input_.lst:
            if not isinstance(inpt_loader,Input_Loader):
                return f"Invalid Input! Should be of format [Input_Loader(1,2,3)...Input_Loader(1,2,3)]"
            
        if size:
            self.connect_all()
        if train:
            output = self.top.train_diff_input(input_)
        else:
            output = self.top.eval_diff_input(input_)
        if not isinstance(output,dict):
            return f"Error Occured! Expected output to be of type: dict but instead received \
            type: {type(output)}"
            
        input_tensor = []
        for (iter,model) in zip(output.items(),self.bottom.models):
            k,v = iter
            input_tensor.append(Input_Loader(model,v[0]))
        
        tensor = Batch(input_tensor)
        if train:
            output = self.bottom.train_diff_input(tensor,override = override)
        else:
            output = self.bottom.eval_diff_input(tensor,override = override)
        
        return output
    
    ''' Immediately goes to second model instead of the first
    by default, it skips through first layer and trains second layer instead'''
    def select_pass(self,input_,train = False,override = False,top_layer = False):
        inpt_tensor = []
        if isinstance(input_,Tensor): 
            for model in self.models:
                inpt_tensor.append(Input_Loader(model,input_))
            input_ = Batch(inpt_tensor)

        for inpt_loader in input_.lst:
            if not isinstance(inpt_loader,Input_Loader):
                return f"Invalid Input! Should be of format [Input_Loader(1,2,3)...Input_Loader(1,2,3)]"
        
        if top_layer:
            if train:
                output = self.bottom.train_diff_input(input_,override = override)
            else: 
                output = self.bottom.eval_diff_input(input_,override = override)

            return output
        else:
            if train:
                output = self.bottom.train_diff_input(input_,override = override)
            else: 
                output = self.bottom.eval_diff_input(input_,override = override)
            
            return output 

