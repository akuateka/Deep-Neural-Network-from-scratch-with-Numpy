import numpy as np
import sys
import time
import copy
import pickle

def print_percent_done(index, total, bar_len=10):
    '''
    index is expected to be 0 based index. 
    0 <= index < total
    '''
    percent_done = (index)/(total-1)*100
    percent_done = round(percent_done, 1)

    done = round(percent_done/(100/bar_len))
    togo = bar_len-done

    done_str = '█'*int(done)
    togo_str = '░'*int(togo)

    print(f'\t⏳: [{done_str}{togo_str}] {index}/{total-1} steps done', end = '\r')

    if round(percent_done) == 100:
        print('\t✅', flush=True)
        
# +++++++++++++++++++++++++++++++++++++++++++++++
#                  DENSE LAYERS
# +++++++++++++++++++++++++++++++++++++++++++++++
# creates Dense layers
class Layer_Dense:
    
    # Layer initialization
    def __init__(self, n_neurons, n_inputs, weight_regularizer_L2 = 0, bias_regularizer_L2 = 0, 
                                            weight_regularizer_L1 = 0, bias_regularizer_L1 = 0):
        
        np.random.seed(0)
        # initialize the Neural Network parameters
        self.weights = 0.01*np.random.randn(n_neurons, n_inputs)     # sometimes : np.sqrt(2/n_inputs)*w
        self.biases = np.zeros((n_neurons, 1))
        
        # set the impact of regularization parameter
        # L2 regularizer
        self.weight_regularizer_L2 = weight_regularizer_L2
        self.bias_regularizer_L2 = bias_regularizer_L2
        
        # L1 regularizer
        self.weight_regularizer_L1 = weight_regularizer_L1
        self.bias_regularizer_L1 = bias_regularizer_L1
        
    # Retrieve layer parameters
    def get_layer_parameters(self):
        return self.weights, self.biases
    
    # Set weights and biases in a layer instance
    def set_layer_parameters(self , weights , biases):
        self.weights = weights
        self.biases = biases
        
    # forward pass trough the network
    def forward(self, inputs, training):
        
        # to remember the inputs
        self.inputs = inputs
        self.output = np.dot(self.weights, inputs) + self.biases
        
    # Backward pass
    def backward(self, dvalues):
        
        # Gradients with respect to the network parameters
        self.dweights = np.dot(dvalues, self.inputs.T)
        self.dbiases = np.sum(dvalues, axis = 1, keepdims=True)
        
        # add the derivate of the L2 term in the gradients of neural network parameters
        # gradient of cost function with respect to the w
        if self.weight_regularizer_L2 > 0:
            self.dweights += 2 * self.weight_regularizer_L2 * self.weights
        
        # gradient of the cost function with respect to the b
        if self.bias_regularizer_L2 > 0:
            self.dbiases += 2 * self.bias_regularizer_L2 * self.biases
            
        # add the derivate of L1 term in the gradients of neural network parameters
        if self.weight_regularizer_L1 > 0:
            
            # derivate of the L1 term
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            
            # gradient of the cost function with respect to the w
            self.dweights += self.weight_regularizer_L1 * dL1
            
        if self.bias_regularizer_L1 > 0:
            
            # derivate of the L1 term
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            
            # gradient of the cost function with respect to the b
            self.dbiases += self.bias_regularizer_L1 * dL1
            
        # Gradients with respect to the inputs values
        self.dinputs = np.dot(self.weights.T, dvalues)
# *******************************************************************************************************

# +++++++++++++++++++++++++++++++++++++++++++++++
#                  ACTIVATION LAYERS
# +++++++++++++++++++++++++++++++++++++++++++++++
# Relu activation function
class Activation_ReLU:
    
    # Forward pass
    def forward(self, inputs, training):
        # remember the inputs
        self.inputs = inputs
        
        # Calculate the output values from inputs
        self.output = np.maximum(0, inputs)
        
    # Backward pass
    def backward(self, dvalues):
        
        # Since we need to make a change on the previous values,
        # let's copy them first
        self.dinputs = dvalues.copy()
        
        # associate a zero gradient where the inputs are less than zero
        self.dinputs[self.inputs <= 0] = 0
        
    # Calculate predictions for outputs
    def predictions(self ,outputs):
        return outputs
    
# Softmax activation
class Activation_Softmax:
    
    # Forward pass
    def forward(self, inputs, training):
        # remember the inputs
        self.inputs = inputs
        
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis = 0, keepdims = True))
        
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis = 0, keepdims = True)
        self.output = probabilities
    
    # Calculate predictions for outputs
    def predictions(self , outputs):
        return np.argmax(outputs, axis = 0)
        
    # Backward pass
    def backward(self , dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(- 1, 1)
            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - \
            np.dot(single_output, single_output.T)
            
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

# Sigmoid activation
class Activation_Sigmoid:
    
    # Forward pass
    def forward(self, inputs, training):
        # save the inputs
        self.inputs = inputs
        self.output = 1/(1 + np.exp(-inputs))
        
    # Calculate predictions for outputs
    def predictions(self , outputs):
        return (outputs > 0.5) * 1
    
    # Backward pass
    def backward(self, dvalues):
        # Gradient with respect to the inputs
        self.dinputs = dvalues * self.output * (1 - self.output)

# Linear activation
class Activation_Linear:
    
    # Forward pass
    def forward(self, inputs, training):
        # Just remember values
        self.inputs = inputs
        self.output = inputs
        
    # Calculate predictions for outputs
    def predictions(self , outputs):
        return outputs
    
    # Backward pass
    def backward(self, dvalues):
        # derivative is 1, 1 * dvalues = dvalues - the chain rule
        self.dinputs = dvalues.copy()
# *******************************************************************************************************        

# +++++++++++++++++++++++++++++++++++++++++++++++
#                  INPUT LAYERS
# +++++++++++++++++++++++++++++++++++++++++++++++
# Input layer
class Layer_Input:
    
    # The forward pass in the input layer 
    def forward(self, inputs, training):
        self.output = inputs
# *******************************************************************************************************    
# +++++++++++++++++++++++++++++++++++++++++++++++
#                  DROPOUT LAYERS
# +++++++++++++++++++++++++++++++++++++++++++++++
# Dropout layer
class Layer_Dropout:
    
    # class constructors
    def __init__(self, rate):
        
        # Store rate, we invert it as for example for dropout
        # of 0.1 we need success rate of 0.9
        self.rate = 1 - rate
        
    # Forward pass
    def forward(self, inputs, training):
        
        # save the inputs
        self.inputs = inputs
        
        if not training:
            self.output = inputs.copy()
            return
        
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size = inputs.shape)/ \
                                        self.rate
        
        # Apply a binary mask over the layer output
        self.output = self.binary_mask * inputs
        
    # Backward pass
    def backward(self, dvalues):
        
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask
# +++++++++++++++++++++++++++++++++++++++++++++++
#                  LOSS FUNCTIONS
# +++++++++++++++++++++++++++++++++++++++++++++++
# Common loss class
class Loss:
    
    # Regularization loss calculation
    def regularization_loss(self):
        
        # 0 by default
        regularization_loss = 0
        
        # Calculate regularization loss
        # iterate all trainable layers
        for layer in self.trainable_layers:
            
            # L1 regularization - weights
            # calculate only when factor greater than 0
            if layer.weight_regularizer_L1 > 0:
                regularization_loss += layer.weight_regularizer_L1 * \
                                            np.sum(np.abs(layer.weights))
            
            # L2 regularization - weights
            if layer.weight_regularizer_L2 > 0 :
                regularization_loss += layer.weight_regularizer_L2 * \
                                            np.sum(layer.weights * layer.weights)
                
            # L1 regularization - biases
            # only calculate when factor greater than 0
            if layer.bias_regularizer_L1 > 0 :
                regularization_loss += layer.bias_regularizer_L1 * \
                                            np.sum(np.abs(layer.biases))

            # L2 regularization - biases
            if layer.bias_regularizer_L2 > 0 :
                regularization_loss += layer.bias_regularizer_L2 * \
                                            np.sum(layer.biases * layer.biases)
        
        return regularization_loss
    
    # Set/remember trainable layers
    def remember_trainable_layers(self , trainable_layers):
        self.trainable_layers = trainable_layers

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self , output , y, *, include_regularization = False):

        # Calculate sample losses
        sample_losses = self.forward(output, y)
        
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        
        # Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        
        # If just data loss - return it
        if not include_regularization:
            return data_loss
        
        # Return the data and regularization losses
        return data_loss, self.regularization_loss()
    
    # Calculates accumulated loss
    def calculate_accumulated(self, * , include_regularization = False):
        
        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count
        
        # If just data loss - return it
        if not include_regularization:
            return data_loss
        
        # Return the data and regularization losses
        return data_loss, self.regularization_loss()

    # Reset variables for accumulated loss
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

# Binary cross-entropy loss
class Loss_BinaryCrossentropy(Loss):
    
    # Forward pass
    def forward(self, y_pred, y_true):
    
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7 , 1 - 1e-7 )
        
        # Calculate sample-wise loss
        sample_losses = - (y_true * np.log(y_pred_clipped) + \
                           ( 1 - y_true) * np.log( 1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis = -1)
        
        # Return losses
        return sample_losses

    # Backward pass
    def backward(self, dvalues, y_true):
    
        # Number of samples
        samples = len(dvalues[0])
        
        # Number of outputs in every sample
        # We'll use the first sample to count them
        n_outputs = len(dvalues)
        
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7 , 1 - 1e-7)
    
        # Calculate gradient
        self.dinputs = - ((y_true / clipped_dvalues) -\
                          ((1 - y_true) / ( 1 - clipped_dvalues))) / n_outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Mean Squared Error loss
class Loss_MeanSquaredError(Loss): # L2 loss
    
    # Forward pass
    def forward(self, y_pred, y_true):
    
        # Calculate loss
        sample_losses = np.mean((y_true - y_pred)**2, axis =-1)
        
        # Return losses
        return sample_losses
    
    # Backward pass
    def backward(self, dvalues, y_true):
        
        # Number of samples
        samples = len(dvalues[0])

        # Number of outputs in every sample
        # We'll use the first sample to count them
        n_outputs = len(dvalues)

        # Gradient on values
        self.dinputs = - 2 * (y_true - dvalues) / n_outputs
        
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Mean Absolute Error loss
class Loss_MeanAbsoluteError(Loss): # L1 loss
    def forward(self, y_pred, y_true):
    
        # Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis =-1)

        # Return losses
        return sample_losses
    
    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues[0])

        # Number of outputs in every sample
        # We'll use the first sample to count them
        n_outputs = len(dvalues)

        # Calculate gradient
        self.dinputs = np.sign(y_true - dvalues) / n_outputs
        
        # Normalize gradient
        self.dinputs = self.dinputs / samples
        
# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    
    # Forward pass
    def forward(self, y_pred, y_true):
        
        # Number of samples in a batch
        samples = len(y_pred[0])
        
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7 , 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1 :
            correct_confidences = y_pred_clipped[y_true, range(samples)]
            
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2 :
            correct_confidences = np.sum(y_pred_clipped*y_true, axis = 0)
        
        # Losses
        negative_log_likelihoods = - np.log(correct_confidences)
        return negative_log_likelihoods
    
    # Backward pass
    def backward(self , dvalues , y_true):

        # Number of samples
        samples = len(dvalues[0])

        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues)
        
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1 :
            y_true = np.eye(labels)[y_true]
        
        # Calculate gradient
        self.dinputs = - y_true / dvalues
        
        # Normalize gradient
        self.dinputs = self.dinputs / samples
        
# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():
    
    # Backward pass
    def backward(self, dvalues, y_true):
        
        # Number of samples
        samples = len(dvalues[0])

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2 :
            y_true = np.argmax(y_true, axis=0)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()

        # Calculate gradient
        self.dinputs[y_true, range(samples)] -= 1
        
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# *******************************************************************************************************

# +++++++++++++++++++++++++++++++++++++++++++++++
#                  OPTIMIZATION ALGORITHMS
# +++++++++++++++++++++++++++++++++++++++++++++++
# Stochastic Gradient Descent
class Optimizer_SGD:
    
    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate = 1.0, decay = 0.001, momentum = 0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iteration = 0
        
    # learning rate decay
    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate*\
                1. / (1. + (self.decay*self.iteration))

    # Update parameters
    def update_params(self, layer):
        
        # If we use momentum
        if self.momentum:
            
            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                
                # Initialize the momentum of layers parameters 
                layer.weight_momentums = np.zeros_like(layer.weights)
                
                # If there is no momentum array for weights
                # The array doesn't exist for biases yet either.
                layer.bias_momentums = np.zeros_like(layer.biases)
                
            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_update = self.momentum*layer.weight_momentums -\
                                (self.current_learning_rate*layer.dweights)
            layer.weight_momentums = weight_update
            
            # build biases update
            bias_update = self.momentum*layer.bias_momentums -\
                                (self.current_learning_rate*layer.dbiases)
            layer.bias_momentums = bias_update
        
        # Vanilla SGD update (without momentum)
        else:
            weight_update = -self.current_learning_rate*layer.dweights
            bias_update = -self.current_learning_rate*layer.dbiases
            
        # Update weights and biases using either
        # vanilla or momentum updates
        layer.weights += weight_update
        layer.biases += bias_update
        
    # Call once after the parameters update
    def post_update_params(self):
        self.iteration += 1

# RMSProp Optimizer
class Optimizer_RMSProp:
    
    # Initialize optimizer - set settings,
    # learning rate of 0.001 is default for this optimizer
    def __init__(self, learning_rate = 0.001, decay = 1e-4, rho = 0.999, epsilon = 1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.rho = rho
        self.epsilon = epsilon
        self.iteration = 0
        
    # learning rate decay
    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate*\
                1. / (1. + (self.decay*self.iteration))

    # Update parameters
    def update_params(self, layer):
            
        # If layer does not contain momentum arrays, create them
        # filled with zeros
        if not hasattr(layer, 'weight_cache'):
                
            # Initialize the momentum of layers parameters 
            layer.weight_cache = np.zeros_like(layer.weights)
                
            # If there is no cache array for weights
            # The array doesn't exist for biases yet either.
            layer.bias_cache = np.zeros_like(layer.biases)
                
        # Update cache with squared current gradients
        layer.weight_cache = self.rho*layer.weight_cache +\
                            ((1 - self.rho)*layer.dweights**2)
            
        layer.bias_cache = self.rho*layer.bias_cache +\
                            ((1 - self.rho)*layer.dbiases**2)
            
        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate*layer.dweights\
                                /(np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate*layer.dbiases\
                                /(np.sqrt(layer.bias_cache) + self.epsilon)
        
    # Call once after the parameters update
    def post_update_params(self):
        self.iteration += 1
        
# ADAM Optimizer
class Optimizer_Adam:
    
    # Initialize optimizer - set settings,
    def __init__(self, learning_rate = 0.001, decay = 0., epsilon = 1e-7, 
                 beta_1 = 0.9, beta_2 = 0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.iteration = 0
        
    # learning rate decay
    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate*\
                1. / (1. + (self.decay*self.iteration))

    # Update parameters
    def update_params(self, layer):
            
        # If layer does not contain momentum arrays, create them
        # filled with zeros
        if not hasattr(layer, 'weight_cache'):
                
            # Initialize the momentum of layers parameters
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
                
            # If there is no cache array for weights
            # The array doesn't exist for biases yet either.
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
            
        # Update momentums with current gradients
        layer.weight_momentums = self.beta_1*layer.weight_momentums +\
                            ((1 - self.beta_1)*layer.dweights)
        layer.bias_momentums = self.beta_1*layer.bias_momentums +\
                            ((1 - self.beta_1)*layer.dbiases)
        
        # Get corrected momentums
        # self.iteration is 0 at first pass and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums /\
                                        (1 - self.beta_1**(self.iteration+1))
        bias_momentums_corrected = layer.bias_momentums /\
                                        (1 - self.beta_1**(self.iteration+1))
                
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2*layer.weight_cache +\
                            ((1 - self.beta_2)*layer.dweights**2)
        layer.bias_cache = self.beta_2*layer.bias_cache +\
                            ((1 - self.beta_2)*layer.dbiases**2)
        
        # Get corrected cache
        # self.iteration is 0 at first pass and we need to start with 1 here
        weight_cache_corrected = layer.weight_cache /\
                                        (1 - self.beta_2**(self.iteration+1))
        bias_cache_corrected = layer.bias_cache /\
                                        (1 - self.beta_2**(self.iteration+1))
        
        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate*weight_momentums_corrected /\
                            (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate*bias_momentums_corrected /\
                            (np.sqrt(bias_cache_corrected) + self.epsilon)
        
    # Call once after the parameters update
    def post_update_params(self):
        self.iteration += 1
# *******************************************************************************************************        

# +++++++++++++++++++++++++++++++++++++++++++++++
#                  ACCURACY EVALUATION
# +++++++++++++++++++++++++++++++++++++++++++++++
# Common accuracy class
class Accuracy:
    
    # Calculates an accuracy
    # given predictions and ground truth values
    def calculate(self, predictions , y):

        # Get comparison results
        comparisons = self.compare(predictions, y)
        
        # Calculate an accuracy
        accuracy = np.mean(comparisons)
        
        # Add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        
        # Return accuracy
        return accuracy
    
    # Calculates accumulated accuracy
    def calculate_accumulated(self):
        # Calculate an accuracy
        accuracy = self.accumulated_sum / self.accumulated_count
        
        # Return the data and regularization losses
        return accuracy

    # Reset variables for accumulated accuracy
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
# Accuracy calculation for regression model
class Accuracy_Regression(Accuracy):
    def __init__(self):
        # Create precision property
        self.precision = None

    # Calculates precision value
    # based on passed in ground truth
    def init(self , y , reinit = False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
        
    # Compares predictions to the ground truth values
    def compare(self , predictions , y):
        return np.abs(predictions - y) < self.precision
    
# Accuracy calculation for classification model
class Accuracy_Categorical(Accuracy):
    # No initialization is needed
    def init(self , y):
        pass
    
    # Compares predictions to the ground truth values
    def compare(self , predictions , y):
        if len (y.shape) == 2 :
            y = np.argmax(y, axis = 0)
        return predictions == y
# *******************************************************************************************************   

# +++++++++++++++++++++++++++++++++++++++++++++++
#                  MODEL DEFINITION
# +++++++++++++++++++++++++++++++++++++++++++++++
# Model Class
class Model:
    
    def __init__(self):
        # create the first list of network object
        self.layers = []
        # Softmax classifier's output object
        self.softmax_classifier_output = None
    
    # Add objects to the model
    def add(self, layer):
        self.layers.append(layer)
        
    # Set the loss and the optimizer of the model
    def set(self, *, loss = None, optimizer = None, accuracy = None):
        if self.loss is not None:    # for the case of network parameters setted
            self.loss = loss
            
        if self.optimizer is not None:    # for the case of network parameters setted
            self.optimizer = optimizer
            
        if self.accuracy is not None:    # for the case of network parameters setted
            self.accuracy = accuracy
        
    # Finalize the model
    def finalize(self):
        
        # Create the input layer to give the property of the 
        # previous layer of the first hidden layer
        self.input_layer = Layer_Input()
        
        # counts the number of layers
        layer_count = len(self.layers)
        
        # Initialize a list containing trainable layers:
        self.trainable_layers = []
        
        for i in range(layer_count):
            
            # Definition of the previous and the next layer
            # If it's the first layer,
            # the previous layer object is the input layer
            if i==0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
                
            # All layers except for the first and the last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
                
            # The last layer - the next object is the loss 
            # Also let's save aside the reference to the last object
            # whose output is the model's output
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
                
            # If layer contains an attribute called "weights",
            # it's a trainable layer -
            # add it to the list of trainable layers
            # We don't need to check for biases -
            # checking for weights is enough
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
                
            # Update loss object with trainable layers
            if self.loss is not None:       # for the case of network parameters setted
                self.loss.remember_trainable_layers(self.trainable_layers)
        
        # If output activation is Softmax and
        # loss function is Categorical Cross-Entropy
        # create an object of combined activation
        # and loss function containing
        # faster gradient calculation
        if isinstance (self.layers[- 1], Activation_Softmax) and \
           isinstance (self.loss, Loss_CategoricalCrossentropy):
            # Create an object of combined activation
            # and loss functions
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()
    
    # Performs forward pass
    def forward(self, X, training):
        
        # Call forward method on the input layer
        # this will set the output property that
        # the first layer in "prev" object is expecting
        self.input_layer.forward(X, training)
        
        # Call forward method of every object in a chain
        # Pass output of the previous object as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
            
        # "layer" is now the last object from the list,
        # return its output
        return layer.output
    
    # Performs backward pass
    def backward(self, output , y):
        
        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # First call backward method
            # on the combined activation/loss
            # this will set dinputs property
            self.softmax_classifier_output.backward(output, y)
            
            # Since we'll not call backward method of the last layer
            # which is Softmax activation
            # as we used combined activation/loss
            # object, let's set dinputs in this object
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            # Call backward method going through
            # all the objects but last
            # in reversed order passing dinputs as a parameter
            for layer in reversed (self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return
        
        # First call backward method on the loss
        # this will set dinputs property that the last
        # layer will try to access shortly
        self.loss.backward(output, y)
        
        # Call backward method going through all the objects
        # in reversed order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
            
    # Train the model
    def fit(self , X , y , * , epochs = 1 , batch_size = None, print_every = 1, 
            validation_data = None):
        
        # Initialize accuracy object
        self.accuracy.init(y)
        
        # Default value if batch size is not being set
        train_steps = 1

        # If there is validation data passed,
        # set default number of steps for validation as well
        
        if validation_data is not None:
            validation_steps = 1
        
            # For better readability
            X_val, y_val = validation_data
        
        # Calculate number of steps
        if batch_size is not None:
            train_steps = len(X[0]) // batch_size
            # Dividing rounds down. If there are some remaining
            # data but not a full batch, this won't include it
            # Add `1` to include this not full batch
            if train_steps * batch_size < len(X[0]):
                train_steps += 1
                
            if validation_data is not None:
                validation_steps = len(X_val[0]) // batch_size
                # Dividing rounds down. If there are some remaining
                # data but nor full batch, this won't include it
                # Add `1` to include this not full batch
                if validation_steps * batch_size < len (X_val[0]):
                    validation_steps += 1
        
        # Main training loop
        for epoch in range(1 , epochs + 1):
                
            # Reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()
            
            # Iterate over steps
            for step in range(train_steps):
                
                # If batch size is not set -
                # train using one step and full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                
                # Otherwise slice a batch
                else :
                    batch_X = X[:, step* batch_size:(step + 1)*batch_size]
                    batch_y = y[0, step* batch_size:(step + 1)*batch_size]
                
                # Perform the forward pass
                output = self.forward(batch_X, training = True)
                
                # Calculate loss
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization = True)
                loss = data_loss + regularization_loss
    
                # Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(output)
                
                accuracy = self.accuracy.calculate(predictions, batch_y)
                
                # Perform backward pass
                self.backward(output, batch_y)
            
                # Optimize (update parameters)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()
                
                # print the steps computations
                if not step % print_every or step == train_steps - 1:
                    print('[===] step: ' + str(step) + '- loss: ' + str(np.round(loss, 4)) +\
                      ' (data_loss: ' + str(np.round(data_loss, 4)) +\
                      ', reg loss: ' + str(np.round(regularization_loss, 3)) + ')' +\
                      ' - accuracy: ' + str(np.round(accuracy, 3)) +\
                      ' - lr: ' + str(np.round(self.optimizer.current_learning_rate, 6))
                         )
                # print the progress bar for the steps
                if batch_size is not None: 
                    print_percent_done(step, train_steps)
                    time.sleep(.0002)
                
            # Get and print epoch loss and accuracy
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization = True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()
            
            # print epoch number
            # if epoch % print_every == 0:
            print('Epoch ' + str(epoch) + '/' + str(epochs))
            print('[=====Training] - loss: ' + str(np.round(epoch_loss, 4)) +\
                  ' (data_loss: ' + str(np.round(epoch_data_loss, 4)) +\
                  ', reg loss: ' + str(np.round(epoch_regularization_loss, 3)) + ')' +\
                  ' - accuracy: ' + str(np.round(epoch_accuracy, 3)) +\
                  ' - lr: ' + str(np.round(self.optimizer.current_learning_rate, 6))
                 )
            
            # If there is the validation data
            if validation_data is not None:
                
                # Reset accumulated values in loss
                # and accuracy objects
                self.loss.new_pass()
                self.accuracy.new_pass()
                
                # Iterate over steps
                for step in range(validation_steps):
                    
                    # If batch size is not set -
                    # train using one step and full dataset
                    if batch_size is None:
                        batch_X = X_val
                        batch_y = y_val
                        
                    # Otherwise slice a batch
                    else :
                        batch_X = X_val[:, step*batch_size:(step + 1)*batch_size]
                        batch_y = y_val[0, step*batch_size:(step + 1)*batch_size]
                
                    # Perform the forward pass
                    output = self.forward(batch_X, training = False)
                
                    # Calculate the loss
                    loss = self.loss.calculate(output, batch_y)
                
                    # Get predictions and calculate an accuracy
                    predictions = self.output_layer_activation.predictions(output)
                    accuracy = self.accuracy.calculate(predictions, batch_y)
                    
                # Get and print validation loss and accuracy
                validation_loss = self.loss.calculate_accumulated()
                validation_accuracy = self.accuracy.calculate_accumulated()
                
                # Print a summary
                print('***** Validation: ' +\
                      ' - validation_loss: ' + str(np.round(validation_loss, 4)) +\
                      ' - validation_accuracy: ' + str(np.round(validation_accuracy, 3))
                     )
                print('\n')
                
    # Evaluates the model using passed in dataset
    def evaluate(self, X_test, y_test, * , batch_size = None):
        
        # Default value if batch size is not being set
        test_steps = 1
        
        # Calculate number of steps
        if batch_size is not None :
            test_steps = len(X_test[0]) // batch_size
            
            # Dividing rounds down. If there are some remaining
            # data, but not a full batch, this won't include it
            # Add `1` to include this not full minibatch
            if test_steps * batch_size < len(X_test[0]):
                test_steps += 1
                
        # Reset accumulated values in loss
        # and accuracy objects
        self.loss.new_pass()
        self.accuracy.new_pass()
        
        # Iterate over steps
        for step in range(validation_steps):
            
            # If batch size is not set -
            # train using one step and full dataset
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            
            # Otherwise slice a batch
            else:
                batch_X = X_val[:, step* batch_size:(step + 1 )* batch_size]
                batch_y = y_val[0, step*batch_size:(step + 1 )*batch_size]
                
            # Perform the forward pass
            output = self.forward(batch_X, training = False)
            
            # Calculate the loss
            self.loss.calculate(output, batch_y)
            
            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)
        
        # Get and print validation loss and accuracy
        test_loss = self.loss.calculate_accumulated()
        test_accuracy = self.accuracy.calculate_accumulated()
        
        # Print a summary
                print('***** Test: ' +\
                      ' - test_loss: ' + str(np.round(test_loss, 4)) +\
                      ' - test_accuracy: ' + str(np.round(test_accuracy, 3))
                     )
                print('\n')
                
    # Retrieves and returns parameters of trainable layers
    def get_parameters(self):
        # Create a list for parameters
        parameters = []
        
        # Iterable trainable layers and get their parameters
        for layer in self.trainable_layers:
            parameters.append(layer.get_layer_parameters())
        
        # Return a list
        return parameters
    
    # Updates the model with new parameters
    def set_parameters(self , parameters):
        
        # Iterate over the parameters and layers
        # and update each layers with each set of the parameters
        for parameter_set, layer in zip (parameters, self.trainable_layers):
            layer.set_layer_parameters(*parameter_set)
            
    # Saves the parameters to a file
    def save_parameters(self , path):
        # Open a file in the binary-write mode
        # and save parameters to it
        
        with open (path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)
            
    # Loads the weights and updates a model instance with them
    def load_parameters(self , path):
        
        # Open file in the binary-read mode,
        # load weights and update trainable layers
        with open (path, 'rb') as f:
            self.set_parameters(pickle.load(f))
            
    # Saves the model
    def save(self , path):
        
        # Make a deep copy of current model instance
        model = copy.deepcopy(self)
        
        # Reset accumulated values in loss and accuracy objects
        model.loss.new_pass()
        model.accuracy.new_pass()

        # Remove data from the input layer
        # and gradients from the loss object
        model.input_layer.__dict__.pop('output' , None)
        model.loss.__dict__.pop('dinputs' , None)

        # For each layer remove inputs, output and dinputs properties
        for layer in model.layers:
            for property in ['inputs' ,'output' ,'dinputs' ,'dweights' ,'dbiases']:
                layer.__dict__.pop(property , None )
        
        # Open a file in the binary-write mode and save the model
        with open (path, 'wb') as f:
            pickle.dump(model, f)
            
    # Loads and returns a model
    @ staticmethod
    def load(path):
        
        # Open file in the binary-read mode, load a model
        with open (path, 'rb') as f:
            model = pickle.load(f)
        
        # Return a model
        return model
    
    # Predicts on the samples
    def predict(self ,X , * , batch_size = None):
        
        # Default value if batch size is not being set
        prediction_steps = 1
        
        # Calculate number of steps
        if batch_size is not None:
            prediction_steps = len(X[0]) // batch_size
            
            # Dividing rounds down. If there are some remaining
            # data, but not a full batch, this won't include it
            # Add `1` to include this not full batch
            
            if prediction_steps*batch_size < len(X[0]):
                prediction_steps += 1
        
        # Model outputs
        output = []
        
        # Iterate over steps
        for step in range(prediction_steps):

            # If batch size is not set -
            # train using one step and full dataset
            if batch_size is None:
                batch_X = X
            
            # Otherwise slice a batch
            else:
                batch_X = X[:, step*batch_size:(step + 1 )*batch_size]
            
            # Perform the forward pass
            batch_output = self.forward(batch_X, training = False)
            
            # Append batch probabilities prediction to the list of predictions
            output.append(batch_output)
        
        # Stack the probabilities predictions and return results
        return np.hstack(output)