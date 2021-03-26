import numpy as np
import math


def sigmoid(x):
    # applying the sigmoid function
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    # computing derivative to the Sigmoid function
    return x * (1 - x)


def relu(x):
    # compute relu
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x < 0, 0, 1)


def linear(x):
    return x


def linear_derivative(x):
    return np.full(x.shape, 1)


def softmax(x):
    ex = np.exp(x)
    return ex / np.sum(ex)


def softmax_derivative(x):
    rx = x.reshape(-1, 1)
    return np.diagflat(rx) - np.dot(rx, rx.T)


def sum_of_squared_error(t, o):
    sub = t - o
    return 0.5 * np.sum(sub**2)


def cross_entropy(pk):
    return -math.log(pk, base=2)

def cross_entropy_derivative():
    pass


activation_functions = {
    # activation_functions
    "relu": relu,
    "sigmoid": sigmoid,
    "softmax": softmax,
    "linear": linear,
}

activation_functions_derivative = {
    # activation_functions_derivative
    "relu": relu_derivative,
    "sigmoid": sigmoid_derivative,
    "softmax": softmax_derivative,
    "linear": linear_derivative
}

error_functions = {
    # error_functions
    "relu": sum_of_squared_error,
    "sigmoid": sum_of_squared_error,
    "softmax": cross_entropy,
    "linear": sum_of_squared_error,
}

class Layer:
    def __init__(self, activation, input, output):
        self.activation_name = activation
        self.activation = activation_functions[activation]
        self.cost_function = error_functions[activation]
        self.activation_derivative = activation_functions_derivative[activation]
        self.W = np.zeros((output, input))
        self.b = np.zeros((output, 1))
        
        self.reset_delta(output)
        self.reset_delta_weight()
        self.reset_delta_bias()
        self.output = np.zeros(output)
        self.net = np.zeros(output)
    
    def set_delta(self, delta):
        self.delta = delta
    
    def set_weight(self, weight):
        self.W = weight

    def set_bias(self, bias):
        self.b = bias

    def add_delta_weight(self, delta):
        self.delta_weight += delta

    def add_delta_bias(self, delta_bias):
        self.delta_bias += delta_bias
    
    def reset_delta(self, output):
        self.delta = np.zeros(output)

    def reset_delta_weight(self):
        self.delta_weight = np.zeros(self.W.shape)

    def reset_delta_bias(self):
        self.delta_bias = np.zeros(self.b.shape)

    def set_output(self, output):
        self.output = output

    def set_net(self, net):
        self.net = net

class NeuralNetwork:
    def __init__(self, learning_rate=0.05, max_iter=500, error_threshold=5, batch_size=5, verbose=False):
        # seeding for random number generation
        np.random.seed(1)
        # converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        self.layers = []
        self.depth = 0
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self.error_threshold = error_threshold

    def add(self, layer):
        self.layers.append(layer)
        self.depth += 1

    def set_params(self, **kwargs):
        self.__dict__.update(kwargs)

    def init_layers(self, layer_description):
        for a in layer_description:
            layer = self.Layer(a.activation_type,
                               a.previous_neuron, a.current_neuron)
            self.add(layer)

    def load_model(self, filename):
        f = open(filename, "r")

        self.depth = int(f.readline())

        for i in range(self.depth):

            n_neuron = int(f.readline())
            activation_type = f.readline()[:-1]
            weight = []

            n_neuron_prev = -1
            for j in range(n_neuron):
                temp = list(map(float, f.readline().split()))
                weight.append(temp)
                if (n_neuron_prev == -1):
                    n_neuron_prev = len(temp)

            layer = self.Layer(activation_type, n_neuron_prev, n_neuron)
            layer.set_weight(np.array(weight))
            bias = np.array(
                list(map(lambda x: [float(x)],
                         f.readline().split())))
            layer.set_bias(bias)

            self.layers.append(layer)

    def save_model(self, filename):
        f = open(filename, "w")

        f.write(str(self.depth) + "\n")
        for layer in self.layers:
            n_neuron = len(layer.b)
            f.write(str(n_neuron) + "\n")
            f.write(layer.activation_name + "\n")
            for i in range(n_neuron):
                f.write(" ".join(list(map(str, layer.W[i]))) + "\n")
            f.write(" ".join(list(map(lambda x: str(x[0]), layer.b))) + "\n")

    def forward_propagate(self, x_inputs):
        a = np.array(x_inputs).T
        for layer in self.layers:
            z = np.dot(layer.W, a) + layer.b
            layer.set_net(z)
            a = layer.activation(z)
            layer.set_output(a)
        return a

    # parameter yang ada -> learning_rate, error_threshold, max_iter, batch_size

    def backward_propagate(self, X_train, y_train, prediction):
        grad = {}

        num_layers = len(self.layers)
        
        print(X_train, y_train)
        print(X_train.shape, y_train.shape)
        
        for i in reversed(range(num_layers)):
            layer = self.layers[i]
            
            # if output layer
            if i == num_layers - 1:
                # use squared error derivative if not softmax
                if self.layers[-1].activation != 'softmax':
                    layer.delta = (prediction - y_train) * layer.activation_derivative(layer.net)
                else:
                    dA = prediction
                    layer.delta = cross_entropy_derivative(prediction, y_train) * layer.activation_derivative(layer.net)
            else:
                next_layer = self.layers[i + 1]
                error = np.dot(next_layer.W.T, next_layer.delta)
                layer.delta = error * layer.activation_derivative(layer.net)
            
            if self.verbose:
                print(f"Layer {i} delta: ")
                print("delta shape: ", layer.delta.shape)
                print(layer.delta)
            
        for i in range(num_layers):
            layer = self.layers[i]
            input_activation = np.atleast_2d(X_train if i == 0 else self.layers[i - 1].output)
            grad["dW" + str(i)] = np.dot(layer.delta, input_activation.T) * self.learning_rate
            grad["db" + str(i)] = layer.delta * self.learning_rate
            if self.verbose:
                print(f"Layer {i} grads: ")
                print("input activation shape: ", input_activation.shape)
                print("Weight shape: ", grad["dW" + str(i)].shape)
                print("bias shape: ", grad["db" + str(i)].shape)
            
        return grad

    def shuffle(self, x_train, y_train):
        sz = len(y_train)
        ids = [i for i in range(sz)]
        np.random.shuffle(ids)
        ret_x, ret_y = [], []
        for i in ids:
            ret_x.append(list(x_train[i]))
            ret_y.append(list(y_train[i]))
        return (ret_x), (ret_y)

    def split_batch(self, x_train, y_train):
        batches_x = []
        batches_y = []

        length = len(x_train)

        for i in range((length // self.batch_size)):
            x_batch = x_train[i * self.batch_size: (i + 1) * self.batch_size]
            y_batch = y_train[i * self.batch_size: (i + 1) * self.batch_size]
            batches_x.append(np.array(x_batch))
            batches_y.append(np.array(y_batch))
        if length % self.batch_size != 0:
            i = length // self.batch_size
            x_batch = x_train[i * self.batch_size:]
            y_batch = y_train[i * self.batch_size:]
            batches_x.append(np.array(x_batch))
            batches_y.append(np.array(y_batch))

        return (batches_x), (batches_y)

    def fit(self, x_train, y_train):
        for iteration in range(self.max_iter):

            x_train, y_train = self.shuffle(x_train, y_train)

            batches_x, batches_y = self.split_batch(x_train, y_train)

            cost_function = 0
            for i in range(len(batches_x)):
                x_input = batches_x[i]
                y_output = batches_y[i]

                prediction = self.forward_propagate(x_input)
                print("ASU", prediction.shape)
                cost_function += self.layers[-1].cost_function(
                    prediction, y_output.T)
                gradients = self.backward_propagate(
                    x_input.T, y_output.T, prediction)
                # update delta phase
                for j, layer in enumerate(self.layers):
                    layer.add_delta_weight(gradients["dW" + str(j)])
                    grad_bias = gradients["db" + str(j)]
                    layer.add_delta_bias(np.sum(grad_bias, axis=1).reshape(len(grad_bias), 1))
                
                # update weights phase
                for j, layer in enumerate(self.layers):
                    layer.W += layer.delta_weight  # gradients["dW" + str(j)]
                    layer.b += layer.delta_bias  # gradients["db" + str(j)]
                
                for j, layer in enumerate(self.layers):
                    layer.reset_delta_weight()
                    layer.reset_delta_bias()

            if cost_function < self.error_threshold:
                break

    def predict(self, x_test):
        prediction = self.forward_propagate(x_test)
        return prediction

    def __str__(self):
        index = 1
        res = ""
        for layer in self.layers:
            res += "{}-th layer\n".format(index)
            res += f"Activation: {layer.activation_name}\n"
            res += "Weight matrix:\n"
            res += layer.W.__str__() + "\n"
            res += "Bias\n"
            res += layer.b.__str__() + "\n\n"
            index += 1
        return res