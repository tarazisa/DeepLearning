import numpy as np
import random as rnd
import math

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

input_values = np.array([0, 3])
layers = np.array([2, 3, 3, 2])
layers_values = {}
weights = {}

def init_layers_values():
    #default all values to 0, do not forget to assign input
    for i in range(layers.size):
        values = np.array([])
        values.resize(layers[i],)
        layers_values[str(i)] = values


def init_weights():

    # Layer 0 is input
    for layer_num in range((layers.size - 1)):
        start = -1 / math.sqrt(layers[layer_num])
        end = 1 / math.sqrt(layers[layer_num])

        from_layer_size = layers[layer_num]
        to_layer_size = layers[layer_num + 1]
        wl = []
        for i in range(to_layer_size):
            w = []
            for node in range(from_layer_size):
                w.append(rnd.uniform(start, end))
            wl.append(w)
        weights[str(layer_num)+"->"+str(layer_num+1)] = wl


def relu(value):
    return max(0., value)


def forward_propagate():
    for layer_num in range(layers.size - 1):
        output_values = []
        values = np.array(layers_values[str(layer_num)])
        w = weights[str(layer_num)+"->"+str(layer_num+1)]
        for node in range(len(w)):
            print ("Now processing weights: %s %s " % (values, w[node]))
            output_value = (values * w[node]).sum()
            output_value = relu(output_value)
            print(output_value)
            output_values.append(output_value)
        layers_values[str(layer_num+1)] = output_values
        print(output_values)
        print(layers_values)


# predict_with_network(input_layer, weights)
init_layers_values()
init_weights()
print (layers_values)
print(weights)


# init Input Values
layers_values['0'] = input_values
forward_propagate()
print (layers_values)
print(weights)
