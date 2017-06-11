import numpy as np
from sklearn.metrics import mean_squared_error

model_output_0 = []
model_output_1 = []

target_outputs = np.array([[12],[12]])
input_data = np.array([[0, 3], [0, 3]])

weights_1 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [1, 1]}

weights_0 = {'node_0': [2, 1],
             'node_1': [1, 3],
             'output': [1, 2]}


def predict_with_network(input_data_row, weights):

    node_0_input = (input_data_row * weights['node_0']).sum()
    node_0_output = relu(node_0_input)

    node_1_input = (input_data_row * weights['node_1']).sum()
    node_1_output = relu(node_1_input)

    hidden_layer_outputs = np.array([node_0_output, node_1_output])
    input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
    return input_to_final_layer


def relu(value):
    return max(value, 0)

for row in input_data:
    model_output_0.append(predict_with_network(row, weights_0))
    model_output_1.append(predict_with_network(row, weights_1))

mse_0 = mean_squared_error(target_outputs, model_output_0)
mse_1 = mean_squared_error(target_outputs, model_output_1)

print(model_output_0)
print(mse_0)
print(model_output_1)
print(mse_1)
