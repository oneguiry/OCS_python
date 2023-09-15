import numpy as np
import math

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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(z):
    sg = sigmoid(z)
    return sg * (1 - sg)


class NeuralNetwork:
    def __init__(self, input_neurons, hidden_layers, count_neurons_layer, y, epochs):
        self.input_neurons = input_neurons
        self.hidden_layers = hidden_layers
        self.count_neurons_layer = count_neurons_layer
        self.y = y
        self.out_put = np.zeros(3)
        self.epochs = epochs

        self.learning_rate = 0.1
        self.W1 = self._init_weights(self.count_neurons_layer)  # Веса от входного слоя к 1му скрытому
        self.W2 = self._init_weights(self.count_neurons_layer)  # Веса от скрытого к выходному
        self.hidden_layer = np.zeros((count_neurons_layer, count_neurons_layer))
        self.act_hidden = np.zeros((count_neurons_layer, count_neurons_layer))
        self.hidden_out = np.zeros((count_neurons_layer, count_neurons_layer))
        self.act_out = np.zeros((count_neurons_layer, count_neurons_layer))

    def _init_weights(self, n):
        matrix = np.zeros((n, n))
        for i in range(0, n):
            matrix[i] = np.random.uniform(low=0.0, high=1.0, size=n)
        return matrix

    def mult_matrix(self, m1, m2):
        result = np.zeros((len(m1), len(m1)))
        for i in range(len(m1)):
            for j in range(len(m1[0])):
                for k in range(len(m2)):
                    print(i, j, k)
                    print(m2)
                    print(m2[k][j])
                    result[i][j] += m1[i][k] * m2[k][j]
        return result

    def mult_vect(self, m1, x):
        res = np.zeros((len(m1), len(m1)))
        for i in range(len(m1)):
            sum = 0
            for j in range(len(m1)):
                print(i, j)
                sum += m1[i][j] * x[j]
        return res

    def subsctract_matrix(self, m1, m2):
        result = np.zeros(len(m1), len(m1))
        for i in range(len(m1)):
            for j in range(len(m1[0])):
                result[i][j] = m1[i][j] - m2[i][j]
        return result

    def add_matrix(self, m1, m2):
        res = np.array(len(m1), len(m1))
        for i in range(len(m1)):
            for j in range(len(m1[0])):
                res[i][j] = m1[i][j] + m2[i][j]
        return res

    def _forward(self, w1, w2, x):
        # TODO: нужно задать смещение
        b = -1
        self.hidden_layer = self.mult_vect(w1, x) - 1
        for i in range(len(self.hidden_layer)):
            for j in range(len(self.hidden_layer[0])):
                print(i, j)
                self.act_hidden[i][j] = sigmoid(self.hidden_layer[i][j])

        self.hidden_out = self.mult_vect(w2, self.hidden_layer) - 1
        for i in range(len(self.hidden_layer)):
            for j in range(len(self.hidden_layer[0])):
                self.act_out[i][j] = sigmoid(self.hidden_out[i][j])

    def _error(self):
        l1_reg = np.abs(self.W2)
        l2_reg = np.sum(self.W1 ** 2)

    def _backward(self):
        sigma3 = self.subsctract_matrix(self.act_out, np.transpose(self.y))
        sigma2 = self.mult_matrix(np.transpose(self.W2), sigma3)
        grad1 = self.mult_matrix(sigma2, self.input_neurons)
        grad2 = self.mult_matrix(sigma3, np.transpose(self.act_out))
        return grad1, grad2

    def _backprog_step(self, x):
        print(x)
        self._forward(self.W1, self.W2, x)
        #grad1, grad2 = self._backward()
        #grad1 = self.add_matrix(grad1, self.W1)
        #grad2 = self.add_matrix(grad2, self.W2)
        #return grad1, grad2

    def fit(self):
        for i in range(self.epochs):
            for figure in range(len(train)):
                self._backprog_step(train[figure])
                #grad1, grad2 = self._backprog_step(train[figure])
                #self.W1 = self.add_matrix(self.W1, grad1)
                #self.W2 = self.add_matrix(self.W2, grad2)


test = [[0, 0, 1],
        [1, 1, 1],
        [1, 0, 1]]
vect = [[0.1, 0.43, -0.21],
        [-1, -0.23, 0.32],
        [0.25, 0.76, -0.99]]

y = [  # 14
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0,
    0, 0, 1, 0, 1, 0, 0,
    0, 1, 0, 0, 0, 1, 0,
    1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0,
]


def main():
    y = [1]
    neural = NeuralNetwork(input_neurons=49,
                           hidden_layers=3,
                           count_neurons_layer=49,
                           y=y,
                           epochs=100)
    neural.fit()


main()
