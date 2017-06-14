import numpy as np
import random as rnd
import math

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

input_layer_dim = 2
hidden_layer_nodes = 2
output_layer_dim = 1

hidden_layers_nodes = np.array([2])
input_values = np.array([0, 3])
layers = np.array([2, 3, 3, 2])
layers_values = {'0': [0, 3]}
output_layer = np.array([0])


weights = {}


def init_layers_values():
    for i in range(layers.size):
        values = np.array([])
        values.resize(layers[i],)
        layers_values[str(i)] = values


def init_weights():
    start = -1/math.sqrt(layers[0])
    end = 1/math.sqrt(layers[0])
    # Weights from input to first hidden layer
    #wl = []
    #for i in range(input_layer.size):
    #    w = []
    #    for layer_size in range(hidden_layers[0]):
    #            w.append(rnd.uniform(start, end))
    #    wl.append(w)
    #weights['i'] = wl

    # Weights from hidden layers up to output
    for layer_num in range((layers.size - 1)):
        from_layer_size = layers[layer_num]
        to_layer_size = layers[layer_num + 1]
        wl = []
        for node in range(from_layer_size):
            w = []
            for i in range(to_layer_size):
                w.append(rnd.uniform(start, end))
            wl.append(w)
        weights[str(layer_num)+"->"+str(layer_num+1)] = wl


# predict_with_network(input_layer, weights)
init_layers_values()
init_weights()
#init Input Values
layers_values['0'] = input_values
print (layers_values)
print(weights)
