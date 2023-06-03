import numpy as np

# creates Dense layers
class Layer_Dense:
    
    # Layer initialization
    def __init__(self, n_neurons, n_inputs, weight_regularizer_L2 = 0, bias_regularizer_L2 = 0, 
                                            weight_regularizer_L1 = 0, bias_regularizer_L1 = 0):
        
        # initialize the Neural Network parameters
        np.random.seed(1234)
        self.weights = np.sqrt(2/n_inputs)*np.random.randn(n_neurons, n_inputs)
        self.biases = np.zeros((n_neurons, 1))
        
        # set the impact of regularization parameter
        # L2 regularizer
        self.weight_regularizer_L2 = weight_regularizer_L2
        self.bias_regularizer_L2 = bias_regularizer_L2
        
        # L1 regularizer
        self.weight_regularizer_L1 = weight_regularizer_L1
        self.bias_regularizer_L1 = bias_regularizer_L1
        
        
    # forward pass trough the network
    def forward(self, inputs):
        
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
            self.biases += self.bias_regularizer_L1 * dL1
            
        # Gradients with respect to the inputs values
        self.dinputs = np.dot(self.weights.T, dvalues)
        
# Relu activation function
class Relu:
    
    # Forward pass
    def forward(self, inputs):
        
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
        
# Softmax activation
class Activation_Softmax:
    
    # Forward pass
    def forward(self , inputs):
        
        # remember the inputs
        self.inputs = inputs
        
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis = 0, keepdims = True))
        
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis = 0, keepdims = True)
        self.output = probabilities
        
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
    def forward(self, inputs):
        
        # save the inputs
        self.inputs = inputs
        self.output = 1/(1 + np.exp(-inputs))
    
    # Backward pass
    def backward(self, dvalues):
        
        # Gradient with respect to the inputs
        self.dinputs = dvalues * self.output * (1 - self.output)

# Linear activation
class Activation_Linear:
    
    # Forward pass
    def forward(self, inputs):
    
        # Just remember values
        self.inputs = inputs
        self.output = inputs

    # Backward pass
    def backward(self, dvalues):
    
    # derivative is 1, 1 * dvalues = dvalues - the chain rule
    self.dinputs = dvalues.copy()

# Common loss class
class Loss:
    # regularization loss
    def regularization_loss(self, layer):
        
        # by default regularization loss = 0
        regularization_loss = 0
        
        # the both regularization loss term implying w and b are added to the initialize 
        # value to compute the regularization loss
        # L2 regularization loss term for weights
        if layer.weight_regularizer_L2 > 0:
            regularization_loss += layer.weight_regularizer_L2 * \
                                            np.sum(layer.weights**2)
        
        # L2 regularization loss term for biases   
        if layer.bias_regularizer_L2 > 0:
            regularization_loss += layer.bias_regularizer_L2 * \
                                            np.sum(layer.biases**2)
        
        # The same process is done for L1 regularization term
        if layer.weight_regularizer_L1 > 0:
            regularization_loss += layer.weight_regularizer_L1 * \
                                            np.sum(np.abs(layer.weights))
        
        # L1 regularization loss term for biases   
        if layer.bias_regularizer_L1 > 0:
            regularization_loss += layer.bias_regularizer_L1 * \
                                            np.sum(np.abs(layer.biases))
        return regularization_loss
    
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self , output , y):
        
        # sample loss computation
        sample_loss = self.forward(output, y)
        
        # calculate the mean losses
        data_loss = np.mean(sample_loss)
        
        # return loss
        return data_loss
    
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
    
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # Forward pass
    def forward(self, inputs, y_true):

        # Output layer's activation function
        self.activation.forward(inputs)

        # Set the output
        self.output = self.activation.output
        
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    
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
        sample_losses = np.mean((y_true - y_pred)**2, axis =- 1)
        
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
class Loss_MeanAbsoluteError (Loss): # L1 loss
    def forward(self, y_pred, y_true):
    
    # Calculate loss
    sample_losses = np.mean(np.abs(y_true - y_pred), axis =- 1)

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
    def __init__(self, learning_rate = 0.001, decay = 1e-5, epsilon = 1e-7, 
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
        
# Dropout layer
class Layer_Dropout:
    
    # class constructors
    def __init__(self, rate):
        
        # Store rate, we invert it as for example for dropout
        # of 0.1 we need success rate of 0.9
        self.rate = 1 - rate
        
    # Forward pass
    def forward(self, inputs):
        
        # save the inputs
        self.inputs = inputs
        
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size = inputs.shape)/ \
                                        self.rate
        
        # Apply a binary mask over the layer output
        self.output = self.binary_mask * inputs
        
    # Backward pass
    def backward(self, dvalues):
        
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask
            
            