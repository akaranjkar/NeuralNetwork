import numpy as np
from nn import NeuralNetwork
from data import Dataset

# Maximum number of iterations. Common to all test data sets
max_iterations = 100000
error_threshold = 0.01
round_digits = 3

print("Maximum number of iterations for all networks is {0}\n"
      "Error threshold for all networks is {1}\n".format(max_iterations,error_threshold))

# Identity - 3 hidden units
input = Dataset.Identity.input
target = Dataset.Identity.target
input_units = input.shape[1]
hidden_units = 3
output_units = target.shape[1]
network_shape = (input_units,hidden_units,output_units)
print("Identity data set with {0} hidden units".format(hidden_units))
print("--------------------------------------\n")
nn1 = NeuralNetwork(network_shape,input,target)
error = None
for i in range(max_iterations):
    e = nn1.backprop()
    error = np.mean(np.abs(e))
    if error < error_threshold:
        print("Desired minimum error reached in iteration {0}".format(i))
        break
print("Error after {0} iterations: {1:0.3f}".format(i, error))
nn1.test(round_digits)
print("Hidden layer weights: ")
print("{0}\n".format(nn1.weights[0]))

# Identity - 4 hidden units
input = Dataset.Identity.input
target = Dataset.Identity.target
input_units = input.shape[1]
hidden_units = 4
output_units = target.shape[1]
network_shape = (input_units,hidden_units,output_units)
print("Identity data set with {0} hidden units".format(hidden_units))
print("--------------------------------------\n")
nn2 = NeuralNetwork(network_shape,input,target)
error = None
for i in range(max_iterations):
    e = nn2.backprop()
    error = np.mean(np.abs(e))
    if error < error_threshold:
        print("Desired minimum error reached in iteration {0}".format(i))
        break
print("Error after {0} iterations: {1:0.3f}".format(i, error))
nn2.test(round_digits)
print("Hidden layer weights: ")
print("{0}\n".format(nn2.weights[0]))

# Tennis
input = Dataset.Tennis.input
target = Dataset.Tennis.target
test_input = Dataset.Tennis.test_input
test_target = Dataset.Tennis.test_target
input_units = input.shape[1]
hidden_units = 5
output_units = target.shape[1]
network_shape = (input_units,hidden_units,output_units)
print("Tennis data set with {0} hidden units".format(hidden_units))
print("--------------------------------------\n")
nn3 = NeuralNetwork(network_shape,input,target,test_input,test_target)
error = None
for i in range(max_iterations):
    e = nn3.backprop()
    error = np.mean(np.abs(e))
    if error < error_threshold:
        print("Desired minimum error reached in iteration {0}".format(i))
        break
print("Error after {0} iterations: {1:0.3f}".format(i, error))
print("\nNetwork output on test data:")
print("(Rounded to {0} decimal points)".format(round_digits))
nn3.test(round_digits)

# Iris
input = Dataset.Iris.input
target = Dataset.Iris.target
test_input = Dataset.Iris.test_input
test_target = Dataset.Iris.test_target
input_units = input.shape[1]
hidden_units = 5
output_units = target.shape[1]
network_shape = (input_units,hidden_units,output_units)
print("\nIris data set with {0} hidden units".format(hidden_units))
print("--------------------------------------\n")
nn4 = NeuralNetwork(network_shape,input,target,test_input,test_target)
error = None
for i in range(max_iterations):
    e = nn4.backprop()
    error = np.mean(np.abs(e))
    if error < error_threshold:
        print("Desired minimum error reached in iteration {0}".format(i))
        break
print("Error after {0} iterations: {1:0.3f}".format(i, error))
print("\nNetwork output on test data:")
print("(Rounded to {0} decimal points)".format(round_digits))
nn4.test(round_digits)

# Iris Noisy
input = Dataset.IrisNoisy.input
target = Dataset.IrisNoisy.target
test_input = Dataset.Iris.test_input
test_target = Dataset.Iris.test_target
input_units = input.shape[1]
hidden_units = 5
output_units = target.shape[1]
network_shape = (input_units,hidden_units,output_units)
print("\nIris noisy data set with {0} hidden units".format(hidden_units))
print("--------------------------------------\n")
nn5 = NeuralNetwork(network_shape,input,target,test_input,test_target)
error = None
for i in range(max_iterations):
    e = nn5.backprop(0.65)
    error = np.mean(np.abs(e))
    if error < error_threshold:
        print("Desired minimum error reached in iteration {0}".format(i))
        break
print("Error after {0} iterations: {1:0.3f}".format(i, error))
print("\nNetwork output on test data:")
print("(Rounded to {0} decimal points)".format(round_digits))
nn5.test(round_digits)