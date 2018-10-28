# -*- coding: utf-8 -*-
"""
- Liner Regression 
- Logistic Regression 
"""

import torch 
import numpy as np 
from torch.autograd import Variable 


class LinearRegressionModel(torch.nn.Module):
    
    def __init__(self,input_size, output_size):
        super(LinearRegressionModel,self).__init__()
        # Applies a linear transformation to the incoming data: y=x*transp(A)+b
        # https://pytorch.org/docs/stable/nn.html#linear 
        self.linear = torch.nn.Linear(input_size,output_size)
        
    def forward(self,x):
        return self.linear(x)
    
class LogisticRegressionModel(torch.nn.Module):
    
    def __init__(self,input_size, output_size):
        super(LogisticRegressionModel,self).__init__()
        # Applies a linear transformation to the incoming data: y=x*transp(A)+b
        # https://pytorch.org/docs/stable/nn.html#linear 
        self.linear = torch.nn.Linear(input_size,output_size)
        
    def forward(self,x):
        return torch.nn.functional.sigmoid(self.linear(x))

def train_model(epochs,x_train,y_train,model,criteria,
                learning_rate=0.01,print_every_n_lines=10):
    optimizer = torch.optim.SGD(model.parameters(),learning_rate,
                                nesterov=True,momentum=0.9,dampening=0)
    
    model.train() # put the model into training mode 
    
    for epoch in range(epochs):
        optimizer.zero_grad() # clear off the gradients from any past operation 
        y_pred = model(x_train) # do the forward pass 
        loss = criteria(y_pred,y_train) # calculate the loss 
        loss.backward() # calculate the loss with the help of back-propagation 
        optimizer.step() # ask the optimizer to adjust parameters based on the gradients 
        if epoch % print_every_n_lines == 0:
            print(epoch,"TRAIN - loss:",float(loss))
    
    model.eval() # put the model into evaluation mode 
    
    return model 

"""
Main 
"""
def main():
    print("torch.__version__:",torch.__version__)
    if torch.cuda.is_available():
        print("GPU Supported",torch.cuda.device_count())
    else:
        print("GPU Not Supported",torch.cuda.device_count())
    print("**** Linear Regression ***")
    # Let's instantiate some data for linear regressrion 
    x_train = Variable(torch.FloatTensor(np.array([[1.01],[2],[3.98],[8.03]])))
    y_train = Variable(torch.FloatTensor([[2.01],[4],[7.99],[16]]))
    
    x_test = Variable(torch.FloatTensor([[10],[20]]))
    #y_test = Variable(torch.Tensor([[20],[40]]))
    
    model = train_model(epochs=900,x_train=x_train,y_train=y_train,
                        model=LinearRegressionModel(1,1),
                        criteria=torch.nn.MSELoss(), ## Mean-Squared-Error loss
                        print_every_n_lines=100)
    test_pred = model(x_test)
    print("Test Prediction (ground truth: 20):",float(test_pred.data[0].numpy()))
    print("Test Prediction (ground truth: 40):",float(test_pred.data[1]))
    
    # Let's check with the normal equation 
    # http://mlwiki.org/index.php/Normal_Equation
    print(">>>> Checking with Normal Equation .... ")
    X = torch.cat((torch.ones(x_train.size()[0],1),x_train),1)
    A = torch.mm(torch.transpose(X, 1, 0),X)
    B = torch.mm(A.inverse(),torch.transpose(X,1,0))
    Theta = torch.mm(B,y_train)
    test_pred_ne = torch.mm(torch.cat((torch.ones(x_test.size()[0],1),x_test),1),
                            Theta)
    print("Test Prediction Normal Equation (ground truth: 20):",float(test_pred_ne.data[0].numpy()))
    print("Test Prediction Normal Equation (ground truth: 40):",float(test_pred_ne.data[1]))
    
    print(">>>> Parameters learnt with the SGD <bias,linear_coeff>")
    print([i[0] for i in model.parameters()]) 
    print(">>>> Parameters from Normal Equation <bias,linear_coeff>")
    print(Theta) 

    print("**** Logistic Regression ***")
    # Let's instantiate some data for linear regressrion 
    x_train = Variable(torch.from_numpy(np.array([[25],[35],[45],[15]])).float())
    y_train = Variable(torch.FloatTensor([[0],[1],[1],[0]]))
    
    x_test = Variable(torch.FloatTensor([[10],[40]]))
    #y_test = Variable(torch.Tensor([[0],[1]]))
    
    model = train_model(epochs=1000,x_train=x_train,y_train=y_train,
                        model=LogisticRegressionModel(1,1),
                        criteria=torch.nn.BCELoss(), ## Binary-Cross-Entropy loss
                        print_every_n_lines=100)
    test_pred = model.forward(x_test)
    print("Test Prediction (ground truth: 0):",float(test_pred.data[0]))
    print("Test Prediction (ground truth: 1):",float(test_pred.data[1].numpy()))
    print("Test Tensor Data:",test_pred.data)
    print("Test Tensor Grad:",test_pred.grad_fn)
    
if __name__ == '__main__':
    main()
    
#torch.__version__: 0.4.1
#GPU Not Supported 0
#**** Linear Regression ***
#0 TRAIN - loss: 99.78645324707031
#100 TRAIN - loss: 0.000623710046056658
#200 TRAIN - loss: 0.0006231185980141163
#300 TRAIN - loss: 0.0006231052684597671
#400 TRAIN - loss: 0.0006231052684597671
#500 TRAIN - loss: 0.0006231052684597671
#600 TRAIN - loss: 0.0006231052684597671
#700 TRAIN - loss: 0.0006231052684597671
#800 TRAIN - loss: 0.0006231052684597671
#Test Prediction (ground truth: 20): 19.942031860351562
#Test Prediction (ground truth: 40): 39.86521911621094
#>>>> Checking with Normal Equation .... 
#Test Prediction Normal Equation (ground truth: 20): 19.942031860351562
#Test Prediction Normal Equation (ground truth: 40): 39.86521911621094
#>>>> Parameters learnt with the SGD <bias,linear_coeff>
#[tensor([1.9923], grad_fn=<SelectBackward>), tensor(0.0188, grad_fn=<SelectBackward>)]
#>>>> Parameters from Normal Equation <bias,linear_coeff>
#tensor([[0.0188],
#        [1.9923]])
#**** Logistic Regression ***
#0 TRAIN - loss: 4.307015419006348
#100 TRAIN - loss: 1.447026252746582
#200 TRAIN - loss: 1.4648678302764893
#300 TRAIN - loss: 0.4446541965007782
#400 TRAIN - loss: 0.2897562086582184
#500 TRAIN - loss: 0.23708854615688324
#600 TRAIN - loss: 0.21650147438049316
#700 TRAIN - loss: 0.20000821352005005
#800 TRAIN - loss: 0.1864228993654251
#900 TRAIN - loss: 0.17498305439949036
#Test Prediction (ground truth: 0): 0.015621320344507694
#Test Prediction (ground truth: 1): 0.9227421283721924
#Test Tensor Data: tensor([[0.0156],
#        [0.9227]])
#Test Tensor Grad: <SigmoidBackward object at 0x118266240>