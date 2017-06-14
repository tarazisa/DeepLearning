import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

model_output = []
targets = []
model_output_1 = []

target_output = np.array([83])
input_data = np.array([0, 3])

weights = {'node_0': [2, 1],
           'node_1': [1, 3],
           'output': [3, 4]}


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


def get_slope(input_data, preds, target):
    # Calculate the gradient slope
    error = preds - target
    slope = 2 * input_data * error
    return slope


def update_weights(weights, slope, learning_rate):
    updated_weights = np.array(weights)
    updated_weights = updated_weights - slope*learning_rate
    return updated_weights


n_updates = 30
mse_hist = []

#for row in input_data:
row = input_data
target = target_output
learn_rate = 0.01
print("Initial weights: %s" % weights)
predict_with_network(row, weights)
for i in range(n_updates):
    prediction = predict_with_network(row, weights)
    s = get_slope(row, prediction, target)
    weights['output'] = update_weights(weights['output'], s, learn_rate)
    targets.append(target)
    model_output.append(prediction)
    mse = mean_squared_error(targets, model_output)
    mse_hist.append(mse)

plt.plot(mse_hist)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.show()

print("Updated Output Weights %s " % weights['output'])
print("Final prediction after %s iterations is %s" % (n_updates, prediction))



