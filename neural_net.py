import numpy as np

train = [  # 1
    [
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
]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(z):
    sg = sigmoid(z)
    return sg * (1 - sg)


class NeuralNetwork:
    def __init__(self, input_neurons, hidden_layers, count_neurons_layer, file_path,
                 epochs: int = 50, ):
        self.input_neurons = input_neurons
        self.hidden_layers = hidden_layers
        self.count_neurons_layer = count_neurons_layer
        self.count_out_neurons = 3
        self.file_path = file_path

        self.learning_rate = 0.1
        self.input_weights = self._init_weights()  # Веса от входного слоя к 1му скрытому
        self.output_weights = self._init_weights()  # Веса от скрытого к выходному
        self.hidden_layer = np.zeros(49)  # Активация скрытого слоя

    def _init_weights(self):
        matrix = np.zeros((49, 49))
        for i in range(0, 49):
            matrix[i] = np.random.uniform(low=0.0, high=1.0, size=49)
        return matrix

    def mult_matrix(self, W, x):
        result = []
        for i in range(len(W)):
            sum = 0
            for j in range(len(W)):
                sum += W[j][i] * x[j]
            result.append(sum)
        return result

    def _forward(self, X, w):
        # TODO: нужно задать смещение
        b = 0
        for i in range(len(X)):
            print(X[i] * w[i])
        # for i in range(0, 49):
        # self.hidden_layer[i] = sigmoid(self.hidden_layer[i])

    def fit(self):
        i = 0
        for figure in range(len(train)):
            print(f'Step #{i}, data={train[figure]}')
            # self._forward(train[figure], self.input_weights)
            break


test = [[1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]]
vect = [1, 2, 3]


def main():
    neural = NeuralNetwork(input_neurons=49,
                           hidden_layers=3,
                           count_neurons_layer=49,
                           file_path="train_data/rect.txt")
    r = neural.mult_matrix(test, vect)
    print(r)
    # neural.fit()


main()
