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
    ret = np.zeros(x.shape)
    for i in range(x.shape[1]):
        ex = np.exp(x[:, i])
        x[:, i] = ex / np.sum(ex)
    return ret

def softmax_derivative(x):
    ret = np.zeros(x.shape);
    for i in range(x.shape[1]):
        j = np.sum(softmax_derivative_util(x[:, i]), axis=1)
        ret[:, i] = j
    return ret


def softmax_derivative_util(x):
    rx = x.reshape(-1, 1)
    return np.diagflat(rx) - np.dot(rx, rx.T)


def sum_of_squared_error(t, o):
    sub = t - o
    return 0.5 * np.sum(sub**2) / o.shape[1]


def cross_entropy(t, o):
    ret = 0
    for i in range(o.shape[1]):
        j = np.argmax(o[:, i])
        ct = clip_scalar(t[j, i])
        ret += -np.log2(ct)
    return ret

lower_threshold = 1e-15
upper_threshold = 1e15
def clip_scalar(x):
    if x < lower_threshold:
        x = lower_threshold
    if x > upper_threshold:
        x = upper_threshold
    return x

def cross_entropy_derivative(t, o):
    ret = o
    for i in range(o.shape[1]):
        j = np.argmax(o[:, i])
        ret[j, i] = -(1-ret[j, i])
    return ret

#gradient clipping to avoid overflow
clip_upper_threshold = 5
clip_lower_threshold = 0.5
def clip(x):
    ret = x
    norm = np.sum(x * x)
    if norm > clip_upper_threshold ** 2:
        ret = ret * (clip_upper_threshold / np.sqrt(norm))
    if norm < clip_lower_threshold ** 2:
        ret = ret * (clip_lower_threshold / np.sqrt(norm))
    ret = np.vectorize(lambda x: 0 if abs(x) < 1e-15 else x)(ret)
    return ret

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
        self.W = np.random.rand(output, input)
        self.b = np.random.rand(output, 1)
        
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
    def __init__(self, learning_rate=0.05, max_iter=500, error_threshold=0.001, batch_size=5, verbose=False):
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
        a = np.array(x_inputs)
        for layer in self.layers:
            z = np.dot(layer.W, a) + layer.b
            layer.set_net(z)
            a = layer.activation(z)
            layer.set_output(a)
            print("testss", a, z, layer.W)
        return a

    # parameter yang ada -> learning_rate, error_threshold, max_iter, batch_size

    def backward_propagate(self, X_train, y_train, prediction):
        grad = {}

        num_layers = len(self.layers)
        
        if self.verbose:
            print(X_train, y_train)
            print(X_train.shape, y_train.shape)
        
        for i in reversed(range(num_layers)):
            layer = self.layers[i]
            
            # if output layer
            if i == num_layers - 1:
                # use squared error derivative if not softmax
                if self.layers[-1].activation_name != 'softmax':
                    layer.delta = clip((prediction - y_train) * clip(layer.activation_derivative(layer.net)))
                else:
                    print("turunan softmax", layer.activation_derivative(layer.net))
                    print("turunan error", cross_entropy_derivative(y_train, prediction))
                    layer.delta = clip(cross_entropy_derivative(y_train, prediction) * clip(layer.activation_derivative(layer.net)))
            else:
                next_layer = self.layers[i + 1]
                error = clip(np.dot(next_layer.W.T, next_layer.delta))
                layer.delta = clip(error * layer.activation_derivative(layer.net))
            
            print("huhuhu", layer.delta)
            
        for i in range(num_layers):
            layer = self.layers[i]
            input_activation = np.atleast_2d(X_train if i == 0 else self.layers[i - 1].output)
            grad["dW" + str(i)] = clip(np.dot(layer.delta, input_activation.T) * self.learning_rate / len(y_train))
            dZ = clip(layer.delta * self.learning_rate)
            grad["db" + str(i)] = clip(np.sum(dZ, axis=1).reshape(len(dZ), 1))
            
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

                prediction = self.forward_propagate(x_input.T)
                print("prediction ", prediction)
                cost_function += self.layers[-1].cost_function(
                    prediction, y_output.T)
                gradients = self.backward_propagate(
                    x_input.T, y_output.T, prediction)
                
                print(f"iteration {iteration}, batch {i}: ")
                print(gradients)
        
                
                # update delta phase
                for j, layer in enumerate(self.layers):
                    layer.add_delta_weight(gradients["dW" + str(j)])
                    layer.add_delta_bias(gradients["db" + str(j)])
                
                # update weights phase
                for j, layer in enumerate(self.layers):
                    layer.W += layer.delta_weight  # gradients["dW" + str(j)]
                    layer.b += layer.delta_bias  # gradients["db" + str(j)]
                
                for j, layer in enumerate(self.layers):
                    layer.reset_delta_weight()
                    layer.reset_delta_bias()
                
                print(self)
        
            cost_function /= len(x_train)
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: error = {cost_function}")
            
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