#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple CNN-based Neural Network Architecture for MNIST
"""

import torch
import torch.cuda as cuda
import torch.nn as nn

from torch.autograd import Variable

# Torchvision module contains various utilities, classes, models and datasets 
# used towards computer vision usecases
from torchvision import datasets
from torchvision import transforms

# Functional module contains helper functions
import torch.nn.functional as F

class MNISTNet(nn.Module):
    
    def __init__(self):
        super().__init__()
               
        # NOTE: All Conv2d layers have a default padding of 0 and stride of 1,
        # which is what we are using.
        
        # Convolution Layer 1                             # 28 x 28 x 1  (input)
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)      # 24 x 24 x 20  (after 1st convolution)
        self.relu1 = nn.ReLU()                            # Same as above
        
        # Convolution Layer 2
        self.conv2 = nn.Conv2d(20, 30, kernel_size=5)     # 20 x 20 x 30  (after 2nd convolution)
        self.conv2_drop = nn.Dropout2d(p=0.5)             # Same as above
        self.maxpool2 = nn.MaxPool2d(2)                   # 10 x 10 x 30  (after pooling)
        self.relu2 = nn.ReLU()                            # Same as above 
        
        # Fully connected layers
        self.fc1 = nn.Linear(3000, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        
        # Convolution Layer 1                    
        x = self.conv1(x)                        
        x = self.relu1(x)                        
        
        # Convolution Layer 2
        x = self.conv2(x)               
        x = self.conv2_drop(x)
        x = self.maxpool2(x)
        x = self.relu2(x)
        
        # Switch from activation maps to vectors
        x = x.view(-1, 3000)
        
        # Fully connected layer 1
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=True)
        
        # Fully connected layer 2
        x = self.fc2(x)
        
        return x
    
    
def train_model(epochs,train_loader,train_ds,test_ds,model,criterion,batch_size,
                learning_rate=0.01,print_every_n_lines=10):

    optimizer = torch.optim.SGD(model.parameters(),learning_rate,
                                nesterov=True,momentum=0.9,dampening=0)    
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    
    for epoch in range(epochs):
        ############################
        # Train
        ############################
        
        iter_loss = 0.0
        correct = 0
        iterations = 0
        
        model.train()                   # Put the network into training mode
        
        for i, (items, classes) in enumerate(train_loader):
            
            # Convert torch tensor to Variable
            items = Variable(items)
            classes = Variable(classes)
            
            # If we have GPU, shift the data to GPU
            if cuda.is_available():
                items = items.cuda()
                classes = classes.cuda()
            
            optimizer.zero_grad()     # Clear off the gradients from any past operation
            outputs = model(items)      # Do the forward pass
            loss = criterion(outputs, classes) # Calculate the loss
            iter_loss += loss.data[0] # Accumulate the loss
            loss.backward()           # Calculate the gradients with help of back propagation
            optimizer.step()          # Ask the optimizer to adjust the parameters based on the gradients
            
            # Record the correct predictions for training data 
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == classes.data).sum()
            iterations += 1
        
        # Record the training loss
        train_loss.append(iter_loss/iterations)
        # Record the training accuracy
        train_accuracy.append((100 * correct / len(train_loader.dataset)))
       
        ############################
        # Validate - How did we do on the unseen dataset?
        ############################
        
        loss = 0.0
        correct = 0
        iterations = 0
    
        model.eval()                    # Put the network into evaluate mode
        
        for i, (items, classes) in enumerate(mnist_valid_loader):
            
            # Convert torch tensor to Variable
            items = Variable(items)
            classes = Variable(classes)
            
            # If we have GPU, shift the data to GPU
            if cuda.is_available():
                items = items.cuda()
                classes = classes.cuda()
            
            outputs = model(items)      # Do the forward pass
            loss += criterion(outputs, classes).data[0] # Calculate the loss
            
            # Record the correct predictions for training data
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == classes.data).sum()
            
            iterations += 1
    
        # Record the validation loss
        valid_loss.append(loss/iterations)
        # Record the validation accuracy
        valid_accuracy.append(correct / len(mnist_valid_loader.dataset) * 100.0)
    
        if epoch % print_every_n_lines == 0 or epoch == (epochs-1):
            print ('Epoch %d/%d, Tr Loss: %.4f, Tr Acc: %.4f, Val Loss: %.4f, Val Acc: %.4f'
               %(epoch+1, num_epochs, train_loss[-1], train_accuracy[-1], 
                 valid_loss[-1], valid_accuracy[-1]))
        
        return model, train_loss, test_loss, train_accuracy, test_accuracy
    
    
"""
Main 
"""
def main():
    start = time.time()
    print("torch.__version__:",torch.__version__)
    if torch.cuda.is_available():
        print("GPU Supported",torch.cuda.device_count())
    else:
        print("GPU Not Supported",torch.cuda.device_count())
    print("**** Simple CNN-based Neural Network Architecture for MNIST ***")
    print(">> Downloading and transforming dataset ... ")
    # Mean and standard deviation of all the pixels in the MNIST dataset
    mean_gray = 0.1307
    stddev_gray = 0.3081
    
    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((mean_gray,), (stddev_gray,))])
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_valid = datasets.MNIST('./data', train=False, download=True, transform=transform)
    batch_size = 1024 # Reduce this if you get out-of-memory error
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_loader = torch.utils.data.DataLoader(mnist_valid, batch_size=batch_size, shuffle=True, num_workers=1)
    ## TRAIN
    model = MNISTNet()
    if cuda.is_available():
        model = model.cuda()
    model, train_loss, test_loss, train_accuracy, 
    test_accuracy = train_model(epochs=500,train_loader=train_loader,
                                train_ds=train_ds,test_ds=test_ds,
                                model=model,
                                criterion=nn.CrossEntropyLoss(),
                                batch_size=batch_size,
                                learning_rate=0.01,print_every_n_lines=100)


    # time 
    print('Time (s): %.2f seconds'%(time.time() - start))
    
if __name__ == '__main__':
    main()

