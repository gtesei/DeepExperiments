#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feed-Forward Neural Networks 
"""

import torch
import torch.nn as nn

from torch.autograd import Variable

import time

# Custom DataSet
from data import iris

class IrisNet(nn.Module):
    
    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
        
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out
    
def train_model(epochs,train_loader,train_ds,test_ds,model,criterion,batch_size,
                learning_rate=0.01,print_every_n_lines=10,use_cuda=False):

    optimizer = torch.optim.SGD(model.parameters(),learning_rate,
                                nesterov=True,momentum=0.9,dampening=0)    
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    
    for epoch in range(epochs):
        model.train() # put the model into training mode 
        train_correct = 0
        train_total = 0
        for i, (items, classes) in enumerate(train_loader):
            # Convert torch tensor to Variable
            if use_cuda:
                items = Variable(items.cuda())
                classes = Variable(classes.cuda())
            else:
                items = Variable(items)
                classes = Variable(classes)
            optimizer.zero_grad()       # Clear off the gradients from any past operation
            outputs = model(items)      # Do the forward pass
            loss = criterion(outputs, classes) # Calculate the loss
            loss.backward()             # Calculate the gradients with help of back propagation
            optimizer.step()            # Ask the optimizer to adjust the parameters based on the gradients
            # Record the correct predictions for training data
            train_total += classes.size(0)    
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == classes.data).sum()
        model.eval() # put the model into evaluation mode 
        # Book keeping
        # Record the loss
        train_loss.append(loss.data[0])
        # What was our train accuracy?
        _train_accuracy = (100 * train_correct / train_total)
        train_accuracy.append(_train_accuracy)
        # How did we do on the test set (the unseen set)
        # Record the correct predictions for test data
        test_items = torch.FloatTensor(test_ds.data.values[:, 0:4])
        test_classes = torch.LongTensor(test_ds.data.values[:, 4])
        if use_cuda:
            test_items = test_items.cuda()
            test_classes = test_classes.cuda()
        outputs = model(Variable(test_items))
        tloss = criterion(outputs, Variable(test_classes))
        test_loss.append(tloss.data[0])
        _, predicted = torch.max(outputs.data, 1)
        total = test_classes.size(0)
        correct = (predicted == test_classes).sum()
        _test_accuracy = (100 * correct / total)
        test_accuracy.append(_test_accuracy)
        if epoch % print_every_n_lines == 0 or epoch == (epochs-1):
                print ('Epoch %d/%d, Train Loss: %.4f, Train Loss: %.4f, Train Acc: %.4f , Test Acc: %.4f ' 
                       %(epoch+1, epochs, loss.data[0], tloss.data[0],_train_accuracy,_test_accuracy))
    
    return model, train_loss, test_loss, train_accuracy, test_accuracy


"""
Main 
"""
def main():
    start = time.time()
    print("torch.__version__:",torch.__version__)
    if torch.cuda.is_available():
        print("GPU Supported",torch.cuda.device_count())
        use_cuda = True 
    else:
        print("GPU Not Supported",torch.cuda.device_count())
        use_cuda = False 
    print("**** Feed-Forward Neural Network ***")
    batch_size = 60
    # Get the datasets
    train_ds, test_ds = iris.get_datasets('data/iris.data.txt')
    # How many instances have we got?
    print('instances in training set: ', len(train_ds))
    print('instances in testing/validation set: ', len(test_ds))
    # Create the dataloaders - for training and validation/testing
    # We will be using the term validation and testing data interchangably
    train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=True)
    # Train 
    model = IrisNet(4, 100, 50, 3) 
    if use_cuda:
        model = model.cuda()
    model, train_loss, test_loss, train_accuracy, 
    test_accuracy = train_model(epochs=500,train_loader=train_loader,
                                train_ds=train_ds,test_ds=test_ds,
                                model=model,
                                criterion=nn.CrossEntropyLoss(),
                                batch_size=batch_size,
                                learning_rate=0.001,print_every_n_lines=100,
                                use_cuda=use_cuda)
    
    # Saving the model to disk, and loading it back
    torch.save(model.state_dict(), "./2.model.pth")
    net2 = IrisNet(4, 100, 50, 3)
    net2.load_state_dict(torch.load("./2.model.pth"))
    output = net2(Variable(torch.FloatTensor([[5.1, 3.5, 1.4, 0.2]])))
    _, predicted_class = torch.max(output.data, 1)
    print('Predicted class: ', predicted_class.numpy()[0])
    print('Expected class: ', 0 )
    print('Time (s): %.2f seconds'%(time.time() - start))
    
if __name__ == '__main__':
    main()
    
# =============================================================================
# torch.__version__: 0.4.1
# GPU Not Supported 0
# **** Feed-Forward Neural Network ***
# instances in training set:  120
# instances in testing/validation set:  30
# Epoch 101/500, Train Loss: 0.3186, Train Loss: 0.3949, Train Acc: 95.0000 , Test Acc: 100.0000 
# Epoch 201/500, Train Loss: 0.1949, Train Loss: 0.2401, Train Acc: 98.0000 , Test Acc: 100.0000 
# Epoch 301/500, Train Loss: 0.1329, Train Loss: 0.1736, Train Acc: 98.0000 , Test Acc: 96.0000 
# Epoch 401/500, Train Loss: 0.0955, Train Loss: 0.1436, Train Acc: 98.0000 , Test Acc: 96.0000 
# Epoch 500/500, Train Loss: 0.1043, Train Loss: 0.1324, Train Acc: 97.0000 , Test Acc: 96.0000 
# Predicted class:  0
# Expected class:  0
# Time (s): 16.76 seconds
# =============================================================================

