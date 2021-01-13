import numpy
import scipy.special


class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learnrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learnrate
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, input_list, target_list):
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.lr * numpy.dot(output_errors * final_outputs * (1 - final_outputs), hidden_outputs.T)
        self.wih += self.lr * numpy.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), inputs.T)

        pass

    def query(self, input_list):
        inputs = numpy.array(input_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

        pass


input_nodes = 784
hidden_nodes = 100
output_node = 10
learning_rate = 0.2

n = neuralNetwork(input_nodes, hidden_nodes, output_node, learning_rate)

training_data_file = open("data/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

test_data_file = open("data/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
scorecard = []
epochs = 3
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_node) + 0.1
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    for test_record in test_data_list:
        test_values = test_record.split(',')
        correct_label = int(test_values[0])
        # print( "correct Label is: ",correct_label)
        test_inputs = (numpy.asfarray(test_values[1:]) / 255.0 * 0.99) + 0.01
        test_outputs = n.query(test_inputs)
        label = numpy.argmax(test_outputs)
        # print("network's answer is:", label)
        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)
            pass

        pass
    scorecard_array = numpy.asfarray(scorecard)
    print("performance is", scorecard_array.sum() / scorecard_array.size)
    pass

# image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
# plt.imshow(image_array, cmap='Greys', interpolation='None')

# scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
