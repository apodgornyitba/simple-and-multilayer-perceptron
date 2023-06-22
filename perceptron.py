from enum import IntEnum
from normalize import feature_scaling, inverse_feature_scaling
import numpy as np

class UpdateMode(IntEnum):
    BATCH = 1
    ONLINE = 2

class ActivationFunc(IntEnum):
    TANH = 1
    LOGISTIC = 2

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.001, max_epochs=100, accepted_error=0.05, update_mode=UpdateMode.ONLINE):
        self.weights = np.zeros(num_inputs + 1)     # Extra for w0 (bias)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.accepted_error = accepted_error
        self.update_mode = update_mode

    def add_bias(self, X):
        if len(X.shape) == 1:  # If X is a 1D array (i.e., a vector)
            X_bias = np.insert(X, 0, 1)
        else:  # If X is a 2D array (i.e., a matrix)
            X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        return X_bias

    def predict(self, input):
        X_bias = self.add_bias(input)
        return self.activation(X_bias)
    
    def activation(self, input):           # Heaviside predict
        value = np.dot(input, self.weights)
        return np.where(value >= 0, 1, -1)

    def train(self, X, y):
        epochs = 0
        errors = []
        bias = np.ones((X.shape[0], 1))
        Xmodified = np.concatenate((bias, X), axis=1) # Add bias to X
    
        while epochs < self.max_epochs:
            match self.update_mode:
                case UpdateMode.ONLINE:
                    for input, solution in zip(Xmodified, y): # zip(X, y) makes tuples of (input, solution) to iterate them
                        prediction = self.activation(input)
                        delta_w = self.delta_w(prediction, solution)
                        self.weights += delta_w * input
                case UpdateMode.BATCH:
                    prediction = self.activation(Xmodified)
                    delta_w = self.delta_w(prediction, y)      
                    self.weights += np.dot(Xmodified.T, delta_w)
            
            actual_error = self.error(Xmodified, y)
            errors.append(actual_error)
            if actual_error < self.accepted_error:
                print("Convergence reached at epoch", epochs)
                break               # Finish by convergence
            epochs += 1

        error = self.error(Xmodified, y)
        errors.append(error)
        print("Epochs: ", epochs)
        return error, errors

    def delta_w(self, predict, solution):
        return self.learning_rate * (solution - predict) 
        
    # Returns the accuracy of the perceptron on the given data. X and y must be the same length
    def error(self, X, y):
        prediction = self.activation(X)
        return 1 - np.mean(prediction == y)       # Checks if all values of prediction and y are equal
    
    def mse_error(self, X, y):
        prediction = self.predict(X)
        return 1 - np.mean(prediction == y)


class SimpleLinealPerceptron(Perceptron):
    # Idendity activation function
    def activation(self, input):                 # input might be a vector or scalar
        value = np.dot(input, self.weights)
        return value
    
    # def train(self, X, y):
    #     epochs = 0
    #     bias = np.ones((X.shape[0], 1))
    #     Xmodified = np.concatenate((bias, X), axis=1) # Add bias to X
    
    #     while epochs < self.max_epochs:
    #         for input, solution in zip(Xmodified, y): # zip(X, y) makes tuples of (input, solution) to iterate them
    #             prediction = self.activation(input)
                
    #             delta_w = self.delta_w(prediction, solution)

    #             self.weights += delta_w * input

    #         prediction = self.activation(Xmodified)
                   
    #         if self.error(Xmodified, y) < self.accepted_error:
    #             print("Convergence reached at epoch", epochs)
    #             break               # Finish by convergence
    #         epochs += 1

    #     print("Epochs: ", epochs)
    #     return self.error(Xmodified, y)

    # Calculates the mean square error of the perceptron on the given data. X and y must be the same length.
    def error(self, X, y):
        prediction = self.activation(X)
        return np.mean((y - prediction)**2)

    def mse_error(self, X, y):
        prediction = self.predict(X)
        return np.mean((y - prediction)**2)


class SimpleNonLinealPerceptron(SimpleLinealPerceptron):
    
    def __init__(self, num_inputs, learning_rate=0.1, max_epochs=100, accepted_error=0.05, update_mode=UpdateMode.BATCH, beta=0.1, activation_func=ActivationFunc.TANH):
        super().__init__(num_inputs, learning_rate, max_epochs, accepted_error, update_mode)
        self.beta = beta
        self.activation_func = activation_func
    
    # tanh Activation function
    def tanh_activation(self, input):
        return np.tanh(self.beta * input)

    # Derivative of tanh activation function
    def tanh_derivative(self, input):
        return self.beta * (1 - np.tanh(self.beta * input)**2)

    # logistic Activation function
    def logistic_activation(self, input):
        return 1 / (1 + np.exp(-2 * self.beta * input))

    # Derivative of logistic activation function
    def logistic_derivative(self, input):
        return 2 * self.beta * self.logistic_activation(input) * (1 - self.logistic_activation(input))
    
    def activation(self, input):
        if self.activation_func == ActivationFunc.TANH:
            dot = np.dot(input, self.weights)
            return self.tanh_activation(dot)
        elif self.activation_func == ActivationFunc.LOGISTIC:
            dot = np.dot(input, self.weights)
            return self.logistic_activation(dot)
        else:
            raise ValueError("Activation function not supported")
    
    def derivative(self, input):
        if self.activation_func == ActivationFunc.TANH:
            return self.tanh_derivative(input)
        elif self.activation_func == ActivationFunc.LOGISTIC:
            return self.logistic_derivative(input)
        else:
            raise ValueError("Activation function not supported")
        
    def train(self, X, y):
        self.yOriginal = y.copy()
        self.ymin = np.min(y)
        self.ymax = np.max(y)
        if self.activation_func == ActivationFunc.TANH:
            y = feature_scaling(y, -1, 1)
        elif self.activation_func == ActivationFunc.LOGISTIC:
            y = feature_scaling(y, 0, 1)

        epochs = 0
        errors = []

        bias = np.ones((X.shape[0], 1))
        Xmodified = np.concatenate((bias, X), axis=1) # Add bias to X

    
        while epochs < self.max_epochs:
            match self.update_mode:
                case UpdateMode.ONLINE:
                    for input, solution in zip(Xmodified, y): # zip(X, y) makes tuples of (input, solution) to iterate them
                        prediction = self.activation(input)
                        delta_w = self.delta_w(prediction, input, solution)
                        self.weights += delta_w * input
                case UpdateMode.BATCH:
                    prediction = self.activation(Xmodified)
                    delta_w = self.delta_w(prediction, Xmodified, y)
                    self.weights += np.dot(Xmodified.T, delta_w)
            
            prediction = self.activation(Xmodified)

            actual_error = self.error(Xmodified)
            errors.append(actual_error)      
            if actual_error < self.accepted_error:
                print("Convergence reached at epoch", epochs)
                break               # Finish by convergence
            epochs += 1

        error = self.error(Xmodified)
        # print(Xmodified)
        # error = self.mse_error(Xmodified[:, 1:], y)
        errors.append(error)
        print("Epochs: ", epochs)
        return error, errors
        
    def delta_w(self, predict, input, solution):
        return self.learning_rate * (solution - predict) * self.derivative(np.dot(input, self.weights))
    
    # Calculates the mean square error of the perceptron on the given data. X and y must be the same length.
    def error(self, X):
        prediction = self.activation(X)
        if self.activation_func == ActivationFunc.TANH:
            prediction = inverse_feature_scaling(prediction, -1, 1, self.ymin, self.ymax)
        elif self.activation_func == ActivationFunc.LOGISTIC:
            prediction = inverse_feature_scaling(prediction, 0, 1, self.ymin, self.ymax)
        return np.mean((self.yOriginal - prediction)**2)
    
    def mse_error(self, X, y):
        prediction = self.predict(X)
        if self.activation_func == ActivationFunc.TANH:
            prediction = inverse_feature_scaling(prediction, -1, 1, self.ymin, self.ymax)
        elif self.activation_func == ActivationFunc.LOGISTIC:
            prediction = inverse_feature_scaling(prediction, 0, 1, self.ymin, self.ymax)
        return np.mean((y - prediction)**2)
    
