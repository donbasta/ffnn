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


activation_functions = {
    # activation_functions
    "relu": relu,
    "sigmoid": sigmoid,
    "softmax": softmax,
    "linear": linear,
}

error_functions = {
    # error_functions
    "relu": sum_of_squared_error,
    "sigmoid": sum_of_squared_error,
    "softmax": cross_entropy,
    "linear": sum_of_squared_error,
}


class NeuralNetwork:
    class Layer:
        def __init__(self, activation, input, output):
            self.activation_name = activation
            self.activation = activation_functions[activation]
            self.activation_derivative = error_functions[activation]
            self.W = np.zeros((output, input))
            self.b = np.zeros((output, 1))

            self.reset_delta()
            self.reset_delta_bias()
            self.output = np.zeros(output)

        def set_weight(self, weight):
            self.W = weight

        def set_bias(self, bias):
            self.b = bias

        def add_delta(self, delta):
            self.delta += delta

        def reset_delta(self):
            self.delta = np.zeros(self.W.shape)

        def add_delta_bias(self, delta_bias):
            self.delta_bias += delta_bias

        def reset_delta_bias(self):
            self.delta_bias = np.zeros(self.b.shape)

        def set_output(self, output):
            self.output = output

    def __init__(self, learning_rate=0.05):
        # seeding for random number generation
        np.random.seed(1)
        # converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        self.layers = []
        self.depth = 0
        self.learning_rate = learning_rate

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

    '''
    1
    2
    sigmoid
    a11 a12 a13
    a21 a22 a23
    b1 b2
    '''

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
            # print(bias)
            # [[-10], [30]]
            # bias.shape = (2, 1)
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
            # print("debug", np.dot(layer.W, a), layer.b)
            # print(layer.W.shape, a.W.shape, layer.b.shape)
            z = np.dot(layer.W, a) + layer.b
            a = layer.activation(z)
            layer.set_output(a)
        return a

    # parameter yang ada -> learning_rate, error_threshold, max_iter, batch_size

    def backward_propagate(self, x_train, y_train, prediction):
        derivatives = {}

        num_layers = len(self.layers)

        # pake sum of squared error dulu biar kebayang, nanti di generalize
        dA = (prediction - y_train)  # dE/doj
        # pake sigmoid dulu biar kebayang
        # dE/dnet_j
        dZ = dA * self.layers[-1].activation_derivative(prediction)
        dW = np.dot(dZ, self.layers[-2].output)  # dE/dW
        db = dZ  # de/db
        dAPrev = np.dot(dZ, self.layers[-1].output)
        derivatives["dW" + str(num_layers-1)] = dW
        derivatives["db" + str(num_layers-1)] = db

        for i in range(len(self.layers) - 2, 0, -1):
            dZ = dA * self.layers[i].activation_derivative(dA)
            dW = np.dot(dZ, dAPrev)
            db = dZ
            dAPrev = np.dot(dZ, self.layers[i].output)
            derivatives["dW" + str(i)] = dW
            derivatives["db" + str(i)] = db

        return derivatives

    def shuffle(self, x_train, y_train):
        sz = len(y_train)
        ids = np.random.shuffle([i for i in range(sz)])
        ret_x, ret_y = [], []
        for i in ids:
            ret_x.append(x_train[i])
            ret_y.append(y_train[i])
        return ret_x, ret_y

    def split_batch(self, x_train, y_train):
        batches_x = []
        batches_y = []

        length = len(x_train)

        for i in range((length // self.batch_size) + 1):
            x_batch = x_train[i * self.batch_size: (i + 1) * self.batch_size]
            y_batch = y_train[i * self.batch_size: (i + 1) * self.batch_size]
            batches_x.append(x_batch)
            batches_y.append(y_batch)

        return batches_x, batches_y

    def fit(self, x_train, y_train):
        for iteration in range(self.max_iter):

            x_train, y_train = self.shuffle(x_train, y_train)

            batches_x, batches_y = self.split_batch(x_train, y_train)

            for j, layer in enumerate(self.layers):
                layer.reset_delta()
                layer.reset_delta_bias()

            cost_function = 0
            for i in range(len(batches_x)):
                x_input = batches_x[i]
                y_output = batches_y[i]
                prediction = self.forward_propagate(x_input)
                cost_function += self.layers[-1].activation_derivative(
                    prediction, y_output)
                gradients = self.backward_propagate(
                    x_train, y_train, prediction)
                # update delta phase
                for j, layer in enumerate(self.layers):
                    layer.add_delta(gradients["dW" + str(j)])
                    layer.add_delta_bias(gradients["db" + str(j)])

            # update weights phase
            for j, layer in enumerate(self.layers):
                layer.W += layer.delta  # gradients["dW" + str(j)]
                layer.b += layer.delta_bias  # gradients["db" + str(j)]

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


if __name__ == "__main__":
    model = NeuralNetwork()

    '''model.load_model("xor-sigmoid.txt")

    print(model)

    print(model.predict([[0, 0], [0, 1], [1, 0], [1, 1]]))'''

    '''model.set_params(learning_rate = 0.05, error_threshold = 0.03, max_iter = 300, batch_size = 5)

    model.init_layers([{"activation_type" : "sigmoid", "previous_neuron" : 3, "current_neuron" : 5}, {"activation_type" : "relu", "previous_neuron" : 5, "current_neuron" : 2}])

    model.fit(x_train, y_train)'''

    model.save_model("out.txt")
