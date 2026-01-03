import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Dense_Layer:

    def __init__(self, inputs, outputs, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.weights = 0.1 * np.random.randn(inputs, outputs)
        self.biases = np.zeros((1, outputs))

        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases

        return self.outputs
    
    def backward(self, dvalues):
        
        

        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.weight_regularizer_l2 * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)
    

class Loss:
    
    def regularization_loss(self, layer):
        # 0 by default
        regularization_loss = 0
        # L1 regularization - weights
        # calculate only when factor greater than 0
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        # L2 regularization - weights
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
        # L1 regularization - biases
        # calculate only when factor greater than 0
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
        # L2 regularization - biases
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss

    def calculate(self, ypred, ytrues):

        #Calculates loss for each sample using Catagorical Cross Entropy Loss
        sample_losses = self.forward(ypred, ytrues)

        #Averages the losses across samples
        data_loss = np.mean(sample_losses)

        return data_loss

class Loss_CategoricalCrossEntropy(Loss):

    def forward(self, ypred, ytrue):

        #number of samples
        samples = len(ypred)

        #prevents log 0
        ypred_clipped = np.clip(ypred, 1e-7, 1 - 1e-7)
        predictions = np.argmax(ypred, axis=1)

        #Singles out the confidence values of the correct catagories
        if len(ytrue.shape) == 1:
            confidences = ypred_clipped[range(samples), ytrue]
            targets = ytrue
        elif len(ytrue.shape) == 2:
            confidences = np.sum(ytrue * ypred_clipped, axis=1)
            targets = np.argmax(ytrue, axis=1)

        
        negative_log = -np.log(confidences)

        #returns a list of the losses for individual samples
        return negative_log



class ReLU_Activation:

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)

        return self.outputs
    
    def backward(self, dvalues):
        
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Softmax_Activation:

    def forward(self, inputs):
        self.inputs = inputs
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        self.outputs = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)

        return self.outputs


class Activation_Softmax_Categorical_CrossEntropy:

    def __init__(self):
        self.activation = Softmax_Activation()
        self.loss = Loss_CategoricalCrossEntropy()

    def forward(self, inputs, ytrue):

        self.activation.forward(inputs)

        self.outputs = self.activation.outputs

        return self.loss.calculate(self.outputs, ytrue)
    
    def backward(self, dvalues, ytrue):

        samples = len(dvalues)

        #Converts ytrue from onehot encoded matrix to list of indicies
        if len(ytrue.shape) == 2:
            ytrue = np.argmax(ytrue, axis=1)

        self.dinputs = dvalues.copy()
        #ypred - ytrue (ytrues are all 1s and 0s)
        self.dinputs[range(samples), ytrue] -= 1
        #Normalize dinputs
        self.dinputs = self.dinputs / samples


class Optimizer_SGD:

    def __init__(self, learning_rate=1.0, decay=0., momentum=0.):
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0
        
    def pre_update_params(self):
        if self.decay:
            self.learning_rate = self.initial_learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if self.momentum:

            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - self.learning_rate * layer.dweights
            bias_updates = self.momentum * layer.bias_momentums - self.learning_rate * layer.dbiases

            layer.weight_momentums = weight_updates
            layer.bias_momentums = bias_updates
        
        else:
            weight_updates = -self.learning_rate * layer.dweights
            bias_updates = -self.learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1


class Optimizer_Adagrad:

    def __init__(self, learning_rate=1.0, decay=0., epsilon=1e-7):
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon

        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.learning_rate = self.initial_learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):

        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        layer.weights += -self.learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class Optimizer_RMSprop:

    def __init__(self, learning_rate=0.02, decay=0., epsilon=1e-7, rho=0.999):
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.rho = rho

        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.learning_rate = self.initial_learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):

        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2

        layer.weights += -self.learning_rate * self.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.learning_rate * self.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

    
class Optimizer_Adam:

    def __init__(self, learning_rate=0.02, decay=0., epsilon=1e-7, beta1=0.9, beta2=0.999):
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.learning_rate = self.initial_learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):

        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta1 * layer.weight_momentums + (1 - self.beta1) * layer.dweights
        layer.bias_momentums = self.beta1 * layer.bias_momentums + (1 - self.beta1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta1**(self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta1**(self.iterations + 1))

        layer.weight_cache = self.beta2 * layer.weight_cache + (1 - self.beta2) * layer.dweights**2
        layer.bias_cache = self.beta2 * layer.bias_cache + (1 - self.beta2) * layer.dbiases**2

        weight_cache_corrected = layer.weight_cache / (1 - self.beta2**(self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta2**(self.iterations + 1))

        layer.weights += -self.learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class Layer_Dropout:

    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs):
        self.inputs = inputs

        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate

        self.outputs = inputs * self.binary_mask

        return self.outputs
    
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

class Network:

    def __init__(self, inputs, shape, outputs, dropout_rate=0., weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.layers = []

        self.shape = shape #(Neurons_per_layer, Layers)
        self.input_neurons = inputs
        self.output_neurons = outputs

        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

        self.initialize_neurons()

        self.dropouts = [Layer_Dropout(dropout_rate)] * (len(self.layers)-1)
        self.hidden_activation = [ReLU_Activation()] * (len(self.layers)-1)
        self.network_activation = Softmax_Activation()
        self.loss_activation = Activation_Softmax_Categorical_CrossEntropy()
        self.loss_function = Loss()

    
    def initialize_neurons(self):
        self.layers.append(Dense_Layer(self.input_neurons, self.shape[0], self.weight_regularizer_l1, self.weight_regularizer_l2, self.bias_regularizer_l1, self.bias_regularizer_l2))

        for i in range(self.shape[1] - 1):
            self.layers.append(Dense_Layer(self.shape[0], self.shape[0]))
        
        self.layers.append(Dense_Layer(self.shape[0], self.output_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        layer_outputs = self.layers[0].forward(inputs)
        final_outputs = self.hidden_activation[0].forward(layer_outputs)

        for i in range(1, len(self.layers)):
            layer_outputs = self.layers[i].forward(final_outputs)
            if i != len(self.layers)-1:
                final_outputs = self.hidden_activation[i].forward(layer_outputs)
                final_outputs = self.dropouts[i].forward(final_outputs)
            else:
                final_outptus = layer_outputs

        self.outputs = self.network_activation.forward(final_outputs)

        return self.outputs

    def forward(self, inputs, ytrues):
        self.inputs = inputs
        layer_outputs = self.layers[0].forward(inputs)
        final_outputs = self.hidden_activation[0].forward(layer_outputs)
        final_outputs = self.dropouts[0].forward(final_outputs)

        for i in range(1, len(self.layers)):
            layer_outputs = self.layers[i].forward(final_outputs)
            if i != len(self.layers)-1:
                final_outputs = self.hidden_activation[i].forward(layer_outputs)
                final_outputs = self.dropouts[i].forward(final_outputs)
                
        
        final_outputs = layer_outputs
        self.loss = self.loss_activation.forward(final_outputs, ytrues)
        self.outputs = self.loss_activation.outputs

        return self.loss_activation.outputs


    
    def train(self, data, ytrues, optimizer, printLoss=False, epochs=10001):

        y = ytrues
        for epoch in range(epochs):

            self.forward(data, ytrues)

            predictions = np.argmax(self.loss_activation.outputs, axis=1)

            if len(y.shape) == 2:
                y = np.argmax(y, axis=1)
            accuracy = np.mean(predictions==y)

            if len(y.shape) == 2:
                y = np.argmax(y, axis=1)

            self.loss_activation.backward(self.loss_activation.outputs, y)
            self.layers[-1].backward(self.loss_activation.dinputs)
            
            self.reg_loss = self.loss_function.regularization_loss(self.layers[-1])

            for i in range(len(self.layers)-2, -1, -1):
                self.dropouts[i].backward(self.layers[i+1].dinputs)
                self.hidden_activation[i].backward(self.dropouts[i].dinputs)
                self.layers[i].backward(self.hidden_activation[i].dinputs)

                self.reg_loss += self.loss_function.regularization_loss(self.layers[i])

            self.loss += self.reg_loss        

            if printLoss and not epoch % 100:
                print(f"epoch: {epoch}, " + f"acc: {accuracy:.3f}, " + f"loss: {self.loss:.3f}, " + f"reg_loss: {self.reg_loss:.3f}")

            

            optimizer.pre_update_params()
            for layer in self.layers:
                optimizer.update_params(layer)
            optimizer.post_update_params()
            #raise Exception


X, y = spiral_data(samples=100, classes=3)

network = Network(2, (64, 1), 3, 0.1, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
optimizer = Optimizer_Adam(learning_rate=0.02, decay=5e-7)

network.train(X, y, optimizer, printLoss=True)
# Create dataset
X, y = spiral_data(samples=1000, classes=3)
network.forward(X, y)
predictions = np.argmax(network.loss_activation.outputs, axis=1)

if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)
print(network.loss, accuracy)
