import numpy as np


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
    return np.where(x <= 0, 0, 1)


def linear(x):
    return x


def linear_derivative(x):
    return np.full(x.shape, 1)


def softmax(x):
    ex = np.exp(x)
    return ex / np.sum(ex)


def softmax_derivative(x):
    pass


activation_functions = {
    # activation_functions
    "relu": relu,
    "sigmoid": sigmoid,
    "softmax": softmax,
    "linear": linear
}


class NeuralNetwork:
    class Layer:
        def __init__(self, activation, input, output):
            self.activation_name = activation
            self.activation = activation_functions[activation]
            self.W = np.zeros((output, input))
            self.b = np.zeros((output, 1))

        def set_weight(self, weight):
            self.W = weight

        def set_bias(self, bias):
            self.b = bias

    def __init__(self):
        # seeding for random number generation
        np.random.seed(1)
        # converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        self.layers = []
        self.depth = 0

    def add(self, layer):
        self.layers.append(layer)
        self.depth += 1

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
                list(map(lambda x: [float(x)], f.readline().split())))
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
            f.write(
                " ".join(list(map(lambda x: str(x[0]), layer.b))) + "\n")

    def forward_propagate(self, x_inputs):
        a = np.array(x_inputs).T
        for layer in self.layers:
            # print("debug", np.dot(layer.W, a), layer.b)
            # print(layer.W.shape, a.W.shape, layer.b.shape)
            z = np.dot(layer.W, a) + layer.b
            a = layer.activation(z)
        return a

    def backward_propagate():
        pass

    def fit(self, x_train, y_train, training_iterations):
        # training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            # siphon the training data via  the neuron
            prediction = self.forward_propagate(x_train)
            # cost_function = error(prediction, y_train)
            self.backward_propagate()

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

    model.load_model("xor-relu-linear.txt")

    print(model)

    print(model.predict([[0, 0], [0, 1], [1, 0], [1, 1]]))

    model.save_model("out.txt")
