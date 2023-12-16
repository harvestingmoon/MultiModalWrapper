from test_models import TestModel
import torch
from layers import GPU_Base
from loaders import Input_Loader,Batch,Adaptor

device = ('cuda:0' if torch.cuda.is_available() else 'cpu ')
'''Testing Ground For Development Of Model'''
model1 = TestModel(6,2)
model2 = TestModel(5,3)
model3 = TestModel(2,1)
model4 = TestModel(3,1)
inpt1,inpt2 = Input_Loader(model1,torch.randn(1,6)), Input_Loader(model2,torch.randn(1,5))
input_tensor = Batch([inpt1,inpt2])
batch_test = GPU_Base("Layer A",model1,model2)
batch_test_2 = GPU_Base("Layer B",model3,model4)
adapt_test = Adaptor(batch_test,batch_test_2)
#print(adapt_test.fwd(input_tensor,train = True))
print(adapt_test.connection())

