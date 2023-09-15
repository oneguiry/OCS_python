import numpy as np

train = [
    [  # 1
        0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 0, ],

    [  # 2
        0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 1, 1, 1, 0,
        0, 1, 0, 0, 0, 1, 0,
        0, 1, 0, 0, 0, 1, 0,
        0, 1, 0, 0, 0, 1, 0,
        0, 1, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 0, ],

    [  # 3
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 0, 1, 0,
        0, 1, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1, 0,
        0, 1, 0, 0, 0, 1, 0,
        0, 1, 0, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 0, ],

    [  # 4
        0, 1, 0, 0, 0, 0, 0,
        0, 1, 1, 1, 1, 1, 0,
        0, 1, 0, 0, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0,
        0, 1, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 0, ],

    [  # 5
        0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 1, 0,
        1, 0, 0, 1, 0, 1, 1,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, ],

    [  # 6
        0, 0, 0, 1, 0, 0, 0,
        0, 0, 1, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 1, 0,
        1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, ],

    [  # 7
        0, 0, 0, 1, 0, 0, 0,
        0, 0, 1, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 1, 0,
        1, 1, 0, 1, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, ],

    [  # 8
        1, 0, 0, 1, 0, 0, 0,
        0, 0, 1, 0, 1, 0, 0,
        0, 1, 0, 1, 1, 1, 0,
        1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, ],

    [  # 9
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0,
        0, 0, 1, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 1, 0, 0,
        0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, ],

    [  # 10
        0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 1, 0, 0, 0,
        0, 0, 1, 0, 1, 0, 0,
        0, 1, 0, 1, 0, 1, 0,
        0, 0, 1, 0, 1, 0, 0,
        0, 0, 0, 1, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, ],

    [  # 11
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0,
        0, 0, 1, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 1, 0, 0,
        0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, ],

    [  # 12
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0,
        0, 0, 1, 0, 1, 0, 0,
        0, 1, 1, 1, 1, 1, 0,
        0, 0, 1, 0, 1, 0, 0,
        0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, ],

    [  # 13
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 0, 0,
        0, 1, 1, 1, 1, 1, 0,
        0, 0, 1, 1, 1, 0, 0,
        0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, ],

    [  # 14
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0,
        0, 0, 1, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 1, 0,
        1, 1, 1, 1, 1, 1, 1, ],

    [  # 15
        0, 0, 1, 1, 1, 0, 0,
        0, 1, 0, 0, 0, 1, 0,
        0, 1, 0, 0, 0, 1, 0,
        0, 1, 0, 0, 0, 1, 0,
        0, 1, 0, 0, 0, 1, 0,
        0, 1, 0, 0, 0, 1, 0,
        0, 0, 1, 1, 1, 0, 0, ],
]


class Neuron:
    def __init__(self, count_input, count_neurons, count_neurons_layer, y, epochs):
        self.input_neurons = count_input
        self.y = y
        self.out_put = np.zeros(3)
        self.epochs = epochs

        self.learning_rate = 0.1
        self._init_weights(count_neurons_layer, count_neurons)
        self.hidden_layers = self._init_hidden_layers(count_neurons_layer, count_neurons)
        self.out_layer = np.zeros(3)

    def _init_weights(self, count_layers, count_neurons):
        self.w = np.zeros((count_layers, count_neurons, count_neurons))
        self.w_input = np.zeros((count_neurons, self.input_neurons))
        for i in range(count_layers):
            for j in range(count_neurons):
                self.w[i][j] = np.random.uniform(low=-1.0, high=1.0, size=count_neurons)
        for i in range(count_neurons):
            self.w_input[i] = np.random.uniform(low=-1.0, high=1.0, size=self.input_neurons)

        self.w_out = np.zeros((count_neurons, 3))
        for i in range(count_neurons):
            self.w_out[i] = np.random.uniform(low=-1.0, high=1.0, size=3)

    def _init_hidden_layers(self, count_layers, count_neurons):
        self.hidden_layers = np.zeros((count_layers, count_neurons))
        for i in range(count_layers):
            self.hidden_layers[i] = np.zeros(count_neurons)
        return self.hidden_layers

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, z):
        sg = self.sigmoid(z)
        return sg * (1 - sg)

    def sigmoid_hidden(self):
        for i in range(len(self.hidden_layers)):
            for j in range(len(self.hidden_layers[0])):
                self.hidden_layers[i][j] = self.sigmoid(self.hidden_layers[i][j])

    def mult_matrix_init(self, data_train):
        vect = np.zeros(len(self.hidden_layers[0]))
        for i in range(len(self.w_input)):
            for j in range(len(self.w_input[0])):
                vect[i] += self.w_input[i][j] * data_train[j]
        return vect

    def mult_layers(self, w, h):
        vect = np.zeros(len(h))
        for i in range(len(w)):
            for j in range(len(h)):
                vect[i] += h[j] * w[i][j]
        return vect

    def _activate_layer(self, vect):
        for i in range(len(vect)):
            vect[i] = self.sigmoid(vect[i])
        return vect

    def _forward(self, j):
        """Активировали первый скрытый слой т.к.
        размерность нейронов на входе отличается от кол-ва нейровнов в скрытом слое
        """
        vect = self.mult_matrix_init(train[j])
        vect2 = self.sigmoid(vect)
        self.hidden_layers[j] = vect2
        """
        Активировали все остальные слои
        """
        for i in range(1, len(self.hidden_layers + 1)):
            vect2 = self.mult_layers(self.w[i - 1], self.hidden_layers[i] - 1)
            self.hidden_layers[i] = self._activate_layer(vect2 - 1)

    def fit(self):
        for i in range(self.epochs):
            for j in range(len(train)):
                self._forward(j)
                break
            break


def main():
    network = Neuron(count_input=49, count_neurons=5, count_neurons_layer=1, y=1, epochs=10)
    print(network.hidden_layers)
    network.fit()
    print(network.hidden_layers)


main()
