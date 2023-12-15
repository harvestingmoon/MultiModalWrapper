from test_models import TestModelGPU,TestModel
import torch
from layers import GPU_Base
from loaders import Input_Loader,Batch

device = ('cuda:0' if torch.cuda.is_available() else 'cpu ')

input_tensor = torch.randn(1,5).to(device)
model = TestModelGPU(5,1)
model = model.to(device)
output = model(input_tensor)

print(output)

model1 = TestModelGPU(6,2)
model2 = TestModelGPU(5,3)
inpt1,inpt2 = Input_Loader(model1,torch.randn(1,6)), Input_Loader(model2,torch.randn(1,5))
input_tensor = Batch([inpt1,inpt2])
batch_test = GPU_Base("Layer A",model1,model2)
print(batch_test.eval_diff_input(input_tensor))
