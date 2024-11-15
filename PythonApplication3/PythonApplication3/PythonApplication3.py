import numpy as np
import neurolab as nl

# exercise 1
np.random.seed(1)
input_team8 = np.random.uniform(-0.6, 0.6, (10, 2))
output_team8 = (input_team8[:, 0] + input_team8[:, 1]).reshape(10, 1)

input_range = [[-0.6, 0.6], [-0.6, 0.6]] 
net = nl.net.newff(input_range, [6, 1])
net.trainf = nl.train.train_gd  

goal = 0.00001
show = 15

training_error = net.train(input_team8, output_team8, show=show, goal=goal)

test_input = np.array([[0.1, 0.2]])
result1 = net.sim(test_input)
print(f"Result 1: {result1}")

# exercise 2
np.random.seed(1)
input_team8 = np.random.uniform(-0.6, 0.6, (10, 2))
output_team8 = (input_team8[:, 0] + input_team8[:, 1]).reshape(10, 1)

input_range = [[-0.6, 0.6], [-0.6, 0.6]]
net = nl.net.newff(input_range, [5, 3, 1])
net.trainf = nl.train.train_gd  

epochs = 1000
show = 100
goal = 0.00001

training_error = net.train(input_team8, output_team8, epochs=epochs, show=show, goal=goal)

test_input = np.array([[0.1, 0.2]])
result2 = net.sim(test_input)
print(f"Result 2: {result2}")

# exercise 3
np.random.seed(1)
input_team8 = np.random.uniform(-0.6, 0.6, (100, 2))
output_team8 = (input_team8[:, 0] + input_team8[:, 1]).reshape(100, 1)

input_range = [[-0.6, 0.6], [-0.6, 0.6]]
net = nl.net.newff(input_range, [6, 1])
net.trainf = nl.train.train_gd

show = 15
goal = 0.00001

training_error = net.train(input_team8, output_team8, show=show, goal=goal)
test_input = np.array([[0.1, 0.2]])
result3 = net.sim(test_input)
print(f"Result 3: {result3}")

# exercise 4
np.random.seed(1)
input_team8 = np.random.uniform(-0.6, 0.6, (100, 2))
output_team8 = (input_team8[:, 0] + input_team8[:, 1]).reshape(100, 1)

input_range = [[-0.6, 0.6], [-0.6, 0.6]]
net = nl.net.newff(input_range, [5, 3, 1])
net.trainf = nl.train.train_gd

epochs = 1000  
show = 100  
goal = 0.00001

training_error = net.train(input_team8, output_team8, epochs=epochs, show=show, goal=goal)
test_input = np.array([[0.1, 0.2]])
result4 = net.sim(test_input)
print(f"Result 4: {result4}")

# exercise 5
np.random.seed(1)
input_team8 = np.random.uniform(-0.6, 0.6, (100, 3))
output_team8 = (input_team8[:, 0] + input_team8[:, 1] + input_team8[:, 2]).reshape(100, 1)

input_range = [[-0.6, 0.6], [-0.6, 0.6], [-0.6, 0.6]]
net = nl.net.newff(input_range, [6, 1])
net.trainf = nl.train.train_gd

show = 15
goal = 0.00001

training_error = net.train(input_team8, output_team8, show=show, goal=goal)
test_input = np.array([[0.2, 0.1, 0.2]])
result5 = net.sim(test_input)
print(f"Result 5: {result5}")

np.random.seed(1)
input_team8 = np.random.uniform(-0.6, 0.6, (100, 3))
output_team8 = (input_team8[:, 0] + input_team8[:, 1] + input_team8[:, 2]).reshape(100, 1)

input_range = [[-0.6, 0.6], [-0.6, 0.6], [-0.6, 0.6]]
net = nl.net.newff(input_range, [5, 3, 1])
net.trainf = nl.train.train_gd

epochs = 1000
show = 100
goal = 0.00001

training_error = net.train(input_team8, output_team8, epochs=epochs, show=show, goal=goal)

test_input = np.array([[0.2, 0.1, 0.2]])
result6 = net.sim(test_input)
print(f"Result 6: {result6}")