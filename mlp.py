
from cmath import nan
from ftplib import error_perm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import random, shuffle
from numpy.linalg import eig
import time, datetime

class MLP:
    
    """ 
    Defines network layers and momentum parameter: 
        num_inputs = input layers (number of features)
        num_hidden = hidden layers
        num_outputs = output layers            
        momentum = momentum value between 0 and 1 
    """
    def __init__(self, num_inputs, num_hidden, num_outputs, momentum):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.momentum = momentum

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]
        
        
        # initiate random weights and bias based on number of network layers
        weights = []
        bias = []
        for i in range(len(layers) - 1):
            w = np.random.default_rng(seed=3).uniform(-2/layers[i-1], 2/layers[i-1], size=(layers[i], layers[i+1]))
            b = np.random.default_rng(seed=3).uniform(-2/layers[i-1], 2/layers[i-1], size=(layers[i+1]))

            weights.append(w)
            bias.append(b)

        self.weights = weights
        self.bias = bias

        # initiate activations based on number of network layers
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)   
        self.activations = activations
        
        # initiate derivatives and deltas based on number of network layers
        derivatives = []
        deltas = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i+1]))
            delta = np.zeros((layers[i+1]))
            derivatives.append(d) 
            deltas.append(delta)
        self.derivatives = derivatives
        self.deltas = deltas

    """ 
    Function to forward propagate and output prediction:
        inputs = training/validation/test feature row (x labels) 
    """
    def forward_propogate(self, inputs):
        activations = inputs
        self.activations[0] = inputs

        for i, w in enumerate(self.weights):
            # calculate net inputs
            net_inputs = np.dot(activations, w) + self.bias[i]

            # pick one activation function : sigmoid or tanh
            activations = self._sigmoid(net_inputs)
#             activations = self.tanh(net_inputs)
            
            # Update/save new activations 
            self.activations[i + 1] = activations

        #return output layer activation (predicted value)
        return activations
    
    """
    Function to back propagate and update derivatives and delta values:
        error = difference between y label and output
    """
    def back_propogate(self, error):
        
        #Loop through derivatives backwards, starting from the end of the neural network
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            
            # pick one activation function : sigmoid or tanh
            delta = error * self._sigmoid_derivative(activations)
#             delta = error * self.dtanh(activations)
            
            delta_reshaped = delta.reshape(delta.shape[0], -1).T # ndarray([0.1, 0.2]) --> ndarray([[0.1, 0.2]])
            current_activations = self.activations[i]
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1) # ndarray([0.1, 0.2]) --> ndarray([[0.1], [0.2]])
            
            # Update derivatives and deltas
            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            self.deltas[i] = delta
            error = np.dot(delta, self.weights[i].T)

        return error

    """
    Function for gradient descent, updating weights and bias values:
        learning_rate = the learning_rate we assign in our train() function
    """
    def grad_descent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            b = self.bias[i]
            delta = self.deltas[i]
            derivatives = self.derivatives[i]
            
            # If using momentum
            if (self.momentum != 0):
                
                # Calculate difference between old and new weights
                old_weights = self.weights[i]
                old_b = self.bias[i]
                
                weights += (derivatives * learning_rate)
                weight_momentum = weights - old_weights
                
                b += (delta * learning_rate)
                b_momentum = b - old_b

                # Update weights and bias values using new equation with momentum
                weights += (self.momentum * weight_momentum) +  (derivatives * learning_rate)
                b += (self.momentum * b_momentum) + (delta * learning_rate)
           
            # Without Momentum
            else:
                # Update weights and bias normally without momentum
                weights += (derivatives * learning_rate)
                b += (delta * learning_rate)
        
    """
    Function for training our data:
        X = training set features
        Y = training set labels
        D = training set dates (to keep track of every training row's date)
        epochs = number of iterations we run through the training set
        learning_rate = value we set for gradient descent
        train_type = training with validation set split, or with full training set.
    """
    def train(self, X, Y, D, epochs, learning_rate, train_type):
        
        # If we want validation set, split data into training and validation sets
        if (train_type == "validation"):
            Xs = np.split(X, [int(.8 * len(X))])
            Ys = np.split(Y, [int(.8 * len(Y))])
            Ds = np.split(D, [int(.8 * len(D))])

            x_train, x_valid = Xs[0], Xs[1]
            y_train, y_valid = Ys[0], Ys[1]
            d_train, d_valid = Ds[0], Ds[1]
            
            valid_error_list = []
        else:
            x_train = X
            y_train = Y
            d_train = D
            
        train_error_list = []
        count = 0
        current_error = 0

        for i in range(epochs):
            
            train_error = 0
            
            # Get new learning rate with Simulated Annealing 
            learning_rate = self.sim_annealing(epochs, i, learning_rate, 0.01)
            
            # iterate through number of rows in our training set
            for x, y in zip(x_train, y_train):
                
                # forward propagation
                output = self.forward_propogate(x)

                # calculate error
                error = y - output

                # back propagation
                self.back_propogate(error)

                # gradient descent
                self.grad_descent(learning_rate)
                
                # mean square error first step
                train_error += self._mse(y, output)
            
            
            # Calculate RMSE and MSE training errors
            rmse_train = np.sqrt(train_error / len(x_train))
            mse_train = train_error / len(x_train)

            # Save training error at current epoch
            train_error_list.append(rmse_train)
            
            # If using validation set
            if (train_type == "validation"):
                valid_error = 0

                # iterate through number of rows in validation set
                for x, y in zip(x_valid, y_valid):

                    # forward propagation
                    output = self.forward_propogate(x)

                    valid_error += self._mse(y, output)
                
                #Calculate RMSE and MSE validation error
                rmse_valid = np.sqrt(valid_error / len(x_valid))
                mse_valid = valid_error / len(x_valid)
                valid_error_list.append(rmse_valid)
                
                # Print validation and training errors every 10 epoch 
                if (count == 10):
                    print('train error: {} , validation error: {} at epoch {}'.format(rmse_train, rmse_valid, i))
                    count = 0
                count += 1
            
                # Stop training if validation error starts increasing
                if (current_error < rmse_valid and current_error!=0):
                    print("Lowest validation error reached.")
                    print("--- RMSE ---")
                    print("Training: {}".format(rmse_train))
                    print("Validation: {}".format(rmse_valid))

                    print("--- MSE ---")                
                    print("Training: {}".format(mse_train))
                    print("Validation: {}".format(mse_valid))

                    print("Epoch {}".format(i))
                    break;
                else:  
                    current_error = rmse_valid
           # If no validation set 
            else:
                # print training error every 10 epoch
                if (count == 10):
                    print('train error: {} at epoch {}'.format(rmse_train, i))
                    count = 0
                count += 1
        
        # If using validation set, plot validation and training errors
        if (train_type == "validation"):
            plt.plot(train_error_list, label='train error', color='blue')
            plt.plot(valid_error_list, label='validation error', color='orange')
            leg = plt.legend(loc='upper right')
            plt.xticks(range(0,len(train_error_list)+1, 100))
            plt.xlabel('Epoch')
            plt.ylabel('Error')
            plt.show()

    """
    Function only used by the test set for predicting results:
        X = test set features
        Y = test set y labels
    """    
    def predict(self, X, Y):
        
        test_error = 0
        test_outputs = []
        print(X)
        # iterate through number of rows in test set
        for x, y in zip(X, Y):
            
            # Get predicted value from forward propagation
            output = self.forward_propogate(x)
            
            # Destandardize value
            output = ((output*(y_train_max - y_train_min) - a)/(b - a)) + y_train_min
            y = ((y*(y_train_max - y_train_min) - a)/(b - a)) + y_train_min
            
            # Save predicted value to list
            test_error += np.average((y - output)**2)
            test_outputs.append(output)

        # Calculate RMSE     
        rmse_test = np.sqrt(test_error / len(x_test))

        # Return RMSE and list of outputs
        return rmse_test, test_outputs 

    """
    Function for Simulated Annealing:
        epochs = number of epochs defined in train() function
        i = current epoch we are on
        learning_rate = value defined in train() function
        p = the end parameter we manually set
    """        
    def sim_annealing(self, epochs, i, learning_rate, p):
        return p + (learning_rate - p) * (1 - (1 / (1 + np.exp(10 - (20 * i / epochs)))))
   

    """
    Function calculating first part of Mean Squared Error:
        target = y label (predictand)
        output = predicted value calculated from forward propagation
    """
    def _mse(self, target, output):  

        # destandardize values
        output = ((output*(y_train_max - y_train_min) - a)/(b - a)) + y_train_min
        target = ((target*(y_train_max - y_train_min) - a)/(b - a)) + y_train_min

        return np.average((target - output)**2)

    """ Sigmoid and Tanh activation functions and there derivative functions:
        x for derivative function: current activation in back propagation
        x for regular function: net inputs calculated in forward propagation"""

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def dtanh(self, x):
        return 1 - x**2







