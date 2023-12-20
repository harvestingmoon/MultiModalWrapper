import torch
import loaders
class Base:
    '''
    This is the base model layer, assuming that all the models are loaded onto CPU
    It can take in any arbitrary number of models and trains them recursively 
    to do: make sure that the models are loaded in parallel (aka parallel processing)
    
    '''
    def __init__ (self, layer_name, *indv_models) -> None:
        self.layer_name = layer_name
        self.models = {model:True for model in indv_models}
        self.indv_models = list(indv_models)
        self.connect = {model: [] for model in indv_models}

    def name(self):
        return self.layer_name
    def __len__(self):
        return len(self.models)
    def get_models(self):
        return self.models
    def switch(self,*off_models): # switch to turn off the models
        for model in off_models:
            if model not in self.models:
                return "Invalid Model"
            else:
                self.models[model] = not self.models[model] 
    
    def add(self,model):
        if model in self.models:
            return f"Model {model} already in layer!"
        else:
            self.models[model] = True
            self.indv_models.append(model)
            self.connect[model] = []

    def remove(self,model):
        if model not in self.models:
            return f"Model {model} not in layer!"
        else:
            self.models.pop(model)
            self.indv_models.remove(model)
            self.connect.pop(model)
            return f"Model {model} completely removed from layer {self.layer_name}"
    
    def indv_connect(self,model_,attach_model):
        if model_ not in self.connect:
            return f"Model {model_} not in layer"
        else:
            self.connect[model_].append(attach_model)


    def train_same_input(self,input):
        outputs = {model:[] for model in self.indv_models}
        try:
            for model in self.models:
                if not self.models[model]:
                    continue
                else:
                    model.train()
                    output = model(input)
                    
                    outputs[model].append(output) # respective outputs
            return outputs
        except: 
            raise Exception("Error From Running Code, either Invalid Tensor shape or error in model")

    
    # input = {"Model class a": tensor input a , "Model class b": tensor input b}
    def train_diff_input(self,inputs,override = False):
        def helper(iter,model):
            for (k,v) in iter.items():
                    if model == k or model == v:
                        return True
            return False

        if not isinstance(inputs,loaders.Batch):
            return "Invalid Input!"
        
        outputs = {model:[] for model in inputs.x}
        nil_override = {model:[] for model in inputs.x}
        try:
            for model in inputs.x: 
                if not helper(self.models,model):
                    return f"Model {model._get_name} not added to {self.layer_name}"
                else: 
                    if model in self.models and override:
                        model.train()
                        output = model(inputs.x[model])
                        outputs[model].append(output)  # this is if override holds true
                    
                    elif model in self.models and not override:
                        if self.models[model] == 0:
                            print(f"Model {model._get_name} is turned off, continuing ...")
                            continue
                        else:
                            model.train()
                            output = model(inputs.x[model])
                            nil_override[model].append(output)
            
            if override:
                return outputs
            else:
                print("Warning! Some models might have been switched off")
                return nil_override

        except:
            raise Exception("Error From Running Code, either Invalid Tensor shape or error in model")
                        
    def eval_same_input(self,input):
        outputs = {model:[] for model in self.indv_models}
        try:
            for model in self.models:
                if not self.models[model]:
                    continue
                else:
                    model.eval()
                    output = model(input)
                    outputs[model].append(output) # respective outputs
            return outputs
        except: 
            raise Exception("Error From Running Code, either Invalid Tensor shape or error in model")
    
    def eval_diff_input(self,inputs,override = False):
        outputs = {model:[] for model in inputs.x}
        nil_override = {model:[] for model in inputs.x}
        def helper(iter,model):
            for (k,v) in iter.items():
                    if model == k or model == v:
                        return True
            return False
        
        if not isinstance(inputs,loaders.Batch):
            return "Invalid Input"
        
        try:
            for model in inputs.x:
                if not helper(self.models,model):
                    return f"Model {model._get_name} not added to {self.layer_name}"
                else:
                    if model in self.models and override:
                        model.eval()
                        output = model(inputs.x[model])
                        outputs[model].append(output)  # this is if override holds true
                    
                    elif model in self.models and not override:
                        if self.models[model] == 0:
                            print(f"Model {model._get_name} is turned off, continuing ...")
                            continue
                        else:
                            model.train()
                            output = model(inputs.x[model])
                            nil_override[model].append(output)
            if override:
                return outputs
            else:
                print("Warning! Some models might have been switched off")
                return nil_override
        except:
            raise Exception("Error From Running Code, either Invalid Tensor shape or")


device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
class GPU_Base(Base):
    '''
     This is the base model layer, assuming that all the models are loaded onto GPU
    It can take in any arbitrary number of models and trains them recursively 
    to do: make sure that the models are loaded in parallel (aka parallel processing)
    
    '''
    def __init__(self, layer_name, *indv_models) -> None: 
        for model in indv_models: 
            model.to(device)
        super().__init__(layer_name, *indv_models)
    def gpu_avail(self):
        if torch.cuda.is_available():
            return "Models are in GPU"
        else: 
            return "Models are not in GPU"
        
    def train_same_input(self,input):
        input = input.to(device)
        return super().train_same_input(input)
    
    def train_diff_input(self, inputs, override=False):
        inputs.is_gpu = not inputs.is_gpu
        for model in inputs.x:
            print(inputs.x[model])
            inputs.x[model] = inputs.x[model].to(device)
        return super().train_diff_input(inputs, override = override)
    
    def eval_same_input(self, input):
        input = input.to(device)
        return super().eval_same_input(input)
    
    def eval_diff_input(self, inputs, override=False):
        inputs.is_gpu = not inputs.is_gpu
        for model in inputs.x:
            inputs.x[model] = inputs.x[model].to(device)
        return super().eval_diff_input(inputs, override  =override)
    