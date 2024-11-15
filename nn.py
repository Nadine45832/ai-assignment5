import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt


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
    input = np.random.uniform(-0.6, 0.6, size=(amount, 2))

    # generate target data
    output = np.array([[els[0] + els[1]] for els in input])

    return input, output


def training_three_input_data(amount=10):
    np.random.seed(1)
    # generate 3 features with values drawn from a uniform distribution between -0.6 and 0.6
    input = np.random.uniform(-0.6, 0.6, size=(amount, 3))

    # generate target data
    output = np.array([[els[0] + els[1] + els[2]] for els in input])

    return input, output


def single_layer_nn(input, output):
    dim_min_1, dim_max_1 = input[:, 0].min(), input[:, 0].max()
    dim_min_2, dim_max_2 = input[:, 1].min(), input[:, 1].max()

    nn = nl.net.newff([[dim_min_1, dim_max_1], [dim_min_2, dim_max_2]], [6, 1])

    errors_progress = nn.train(input, output, show=15, goal=0.00001)

    plot_errors(errors_progress)

    print(nn.sim([[0.1, 0.2]]))


def three_input_single_layer_nn(input, output):
    dim_min_1, dim_max_1 = input[:, 0].min(), input[:, 0].max()
    dim_min_2, dim_max_2 = input[:, 1].min(), input[:, 1].max()
    dim_min_3, dim_max_3 = input[:, 2].min(), input[:, 2].max()

    nn = nl.net.newff([[dim_min_1, dim_max_1], [dim_min_2, dim_max_2], [dim_min_3, dim_max_3]], [6, 1])

    errors_progress = nn.train(input, output, show=15, goal=0.00001)

    plot_errors(errors_progress)

    print(nn.sim([[0.2, 0.1, 0.2]]))


def multi_layer_nn(input, output):
    dim_min_1, dim_max_1 = input[:, 0].min(), input[:, 0].max()
    dim_min_2, dim_max_2 = input[:, 1].min(), input[:, 1].max()

    nn = nl.net.newff([[dim_min_1, dim_max_1], [dim_min_2, dim_max_2]], [5, 3, 1])

    nn.trainf = nl.train.train_gd

    errors_progress = nn.train(input, output, show=100, goal=0.00001, epochs=1000)

    plot_errors(errors_progress)

    print(nn.sim([[0.1, 0.2]]))


def three_input_multi_layer_nn(input, output):
    dim_min_1, dim_max_1 = input[:, 0].min(), input[:, 0].max()
    dim_min_2, dim_max_2 = input[:, 1].min(), input[:, 1].max()
    dim_min_3, dim_max_3 = input[:, 2].min(), input[:, 2].max()

    nn = nl.net.newff([[dim_min_1, dim_max_1], [dim_min_2, dim_max_2], [dim_min_3, dim_max_3]], [5, 3, 1])

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
