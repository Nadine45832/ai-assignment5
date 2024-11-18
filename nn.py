import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt

MIN_VALUE = -0.6
MAX_VALUE = 0.6
MIN_MAX_RANGE = [-0.6, 0.6]


def plot_errors(error_progress):
    plt.figure()
    plt.plot(error_progress)
    plt.xlabel('Number of epochs')
    plt.ylabel('Error')
    plt.grid()
    plt.show()


def training_data(amount=10):
    np.random.seed(1)
    # generate 2 features with values drawn from a uniform distribution between -0.6 and 0.6
    input = np.random.uniform(MIN_VALUE, MAX_VALUE, size=(amount, 2))

    # generate target data
    output = np.array([[els[0] + els[1]] for els in input])

    return input, output


def training_three_input_data(amount=10):
    np.random.seed(1)
    # generate 3 features with values drawn from a uniform distribution between -0.6 and 0.6
    input = np.random.uniform(MIN_VALUE, MAX_VALUE, size=(amount, 3))

    # generate target data
    output = np.array([[els[0] + els[1] + els[2]] for els in input])

    return input, output


def single_layer_nn(input, output):
    nn = nl.net.newff([MIN_MAX_RANGE, MIN_MAX_RANGE], [6, 1])

    # nn.trainf = nl.train.train_gd
    # print(nn.trainf)

    errors_progress = nn.train(input, output, show=15, goal=0.00001, epochs=1000)

    plot_errors(errors_progress)

    print(nn.sim([[0.1, 0.2]]))


def three_input_single_layer_nn(input, output):
    nn = nl.net.newff([MIN_MAX_RANGE, MIN_MAX_RANGE, MIN_MAX_RANGE], [6, 1])

    errors_progress = nn.train(input, output, show=15, goal=0.00001)

    plot_errors(errors_progress)

    print(nn.sim([[0.2, 0.1, 0.2]]))


def multi_layer_nn(input, output):
    nn = nl.net.newff([MIN_MAX_RANGE, MIN_MAX_RANGE], [5, 3, 1])

    nn.trainf = nl.train.train_gd

    errors_progress = nn.train(input, output, show=100, goal=0.00001, epochs=1000)

    plot_errors(errors_progress)

    print(nn.sim([[0.1, 0.2]]))


def three_input_multi_layer_nn(input, output):
    nn = nl.net.newff([MIN_MAX_RANGE, MIN_MAX_RANGE, MIN_MAX_RANGE], [5, 3, 1])

    nn.trainf = nl.train.train_gd

    errors_progress = nn.train(input, output, show=100, goal=0.00001, epochs=1000)

    plot_errors(errors_progress)

    print(nn.sim([[0.2, 0.1, 0.2]]))


def main():
    # Single layer feed forward to recognize sum
    input_team8, output_team8 = training_data()
    single_layer_nn(input_team8, output_team8)
    multi_layer_nn(input_team8, output_team8)

    input_team8, output_team8 = training_data(100)
    single_layer_nn(input_team8, output_team8)
    multi_layer_nn(input_team8, output_team8)

    input_team8, output_team8 = training_three_input_data()
    three_input_single_layer_nn(input_team8, output_team8)

    input_team8, output_team8 = training_three_input_data(100)
    three_input_multi_layer_nn(input_team8, output_team8)


if __name__ == "__main__":
    main()
