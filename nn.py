import numpy as np

def sigmoid(x, derivative = False):
    if not derivative:
        return 1 / (1 + np.exp(-x))
    else:
        out = sigmoid(x)
        return out * (1 - out)

# np.random.seed(1) # For testing

class NeuralNetwork:

    def __init__(self,shape,input,target,test_input = None,test_target=None):
        self.shape = shape
        self.size = len(shape)
        self.weights = []
        self.learning_rate = 0.01
        self.initialize_weights()
        self.transfer_func = sigmoid
        self.input = input
        self.target = target
        self.test_input = test_input
        self.test_target = test_target
        if test_input is None or test_target is None:
            self.test_input = input
            self.test_target = target
        self.initialize_temp_matrices(self.input.shape[0])
        self.initialize_delta_matrix(self.input.shape[0])


    def initialize_weights(self):
        for i in range(self.size):
            if i == self.size - 1:
                self.weights.append(None)
            else:
                self.weights.append(np.random.normal(size=[self.shape[i] + 1, self.shape[i + 1]],scale=1E-3))

    def initialize_temp_matrices(self,batch):
        self._layer_output = []
        self._layer_input = []
        self._derivative = []
        self._prev_gradient = []
        # Matrix to hold output values
        for i in range(self.size):
            if i == self.size - 1:
                self._layer_output.append(np.zeros((batch, self.shape[i])))
            else:
                self._layer_output.append(np.zeros((batch, self.shape[i] + 1)))
        # Matrix holding inputs
        for i in range(self.size):
            if i == 0:
                self._layer_input.append(None)
            elif i == self.size - 1:
                self._layer_input.append(np.zeros((batch, self.shape[i])))
            else:
                self._layer_input.append(np.zeros((batch, self.shape[i] + 1)))
        # Matrix that holds derivatives of sigmoid function
        for i in range(self.size):
            if i == 0:
                self._derivative.append(None)
            elif i == self.size - 1:
                self._derivative.append(None)
            else:
                self._derivative.append(np.zeros((self.shape[i] + 1, batch)))
        # Matrix to hold the previous gradients
        for i in range(self.size-1):
            self._prev_gradient.append(None)

    def initialize_delta_matrix(self,batch):
        self._deltas = []
        for i in range(self.size):
            if i == 0:
                self._deltas.append(None)
            elif i == self.size - 1:
                self._deltas.append(np.zeros((batch, self.shape[i])))
            else:
                self._deltas.append(np.zeros((batch, self.shape[i] + 1)))

    def forward(self,input):
        batch = input.shape[0]
        for i in range(self.size):
            if i == 0:
                # Output of the input layer is just a matrix of the inputs with an additional column for the bias
                self._layer_output[i] = np.append(input, np.ones((batch, 1)), axis=1)
            elif i == self.size - 1:
                # Take output of previous layer and dot product with weights
                # Apply sigmoid and output
                self._layer_input[i] = self._layer_output[i - 1].dot(self.weights[i - 1])
                self._layer_output[i] = sigmoid(self._layer_input[i])
            else:
                # Take output of previous layer and dot product with weights
                # Apply sigmoid and append bias column
                self._layer_input[i] = self._layer_output[i - 1].dot(self.weights[i - 1])
                self._layer_output[i] = sigmoid(self._layer_input[i])
                self._layer_output[i] = np.append(self._layer_output[i], np.ones((batch, 1)), axis=1)
                # Also calculate and store sigmoid derivative for use in back propagation
                self._derivative[i] = sigmoid(self._layer_input[i], True).T
        return self._layer_output[-1]

    def backprop(self,momentum = 0.5):
        # Do forward propagation
        self.forward(self.input)
        # Error is output - target
        output_error = self._layer_output[-1] - self.target
        # Take transpose of deltas of output layer
        self._deltas[-1] = (self._layer_output[-1] - self.target).T
        for i in range(self.size - 2,0,-1):
            # Don't consider weights of bias
            weights_nobias = self.weights[i][0:-1, :]
            # Calculate deltas for the hidden layers
            # Delta is weights dot product deltas * derivative
            self._deltas[i] = weights_nobias.dot(self._deltas[i+1]) * self._derivative[i]

        for i in range(self.size - 1):
            # Calculate momentum component if this isn't the first run
            momentum_component = 0
            if self._prev_gradient[i] is not None:
                momentum_component = momentum * self._prev_gradient[i]
            # Adjust weights
            weights_gradient = -self.learning_rate * (self._deltas[i+1].dot(self._layer_output[i])).T
            self.weights[i] +=weights_gradient + momentum_component
            self._prev_gradient[i] = weights_gradient

        return output_error

    def test(self,round_digits=0):
        self.initialize_temp_matrices(self.test_input.shape[0])
        success = 0
        test_output = self.forward(self.test_input)
        for i in range(self.test_input.shape[0]):
            error = np.mean(np.abs(self.test_target[i] - np.round(test_output[i])))
            if error == 0:
                success += 1
            print("\nInput: {0}\n"
                  "Target: {1} Network output: {2}\n"
                  .format(self.test_input[i],self.test_target[i],np.round(test_output[i],round_digits)))
        print("Accuracy: {0}%".format(float(success)/self.test_input.shape[0] * 100))
