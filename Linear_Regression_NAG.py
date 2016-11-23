"""Tutorial to implement (S)GD on linear regression using basic python
code and theano.py"""

#### Libraries
# Standard Libraries
from collections import OrderedDict

# Third Party Libraries
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T

# Paramater iniitilization
# Symbolic variable
X = T.matrix(name='X', dtype=theano.config.floatX)
y = T.vector(name='y', dtype=theano.config.floatX)
lr = T.scalar(name='learn_rate', dtype=theano.config.floatX)

# Variables that will be updated, hence are declared as `theano.share`
theta = theano.shared(name='theta', 
					  value=np.random.uniform(-10, 10, size=(3))
					  	.astype(theano.config.floatX))
bias = theano.shared(name='bias',
					 value=np.random.uniform(0, 20, size=(1, 1))
						.astype(theano.config.floatX),
					 broadcastable=(True, True))

# Momentum parameters
mu = T.scalar(name='momentum', dtype=theano.config.floatX)
v_theta = theano.shared(name='velocity_theta', 
						value=np.zeros(3, dtype=theano.config.floatX))
v_bias = theano.shared(name='velocity_bias',
					   value=np.zeros((1, 1), dtype=theano.config.floatX),
					   broadcastable=(True, True))

params = [theta, bias]
vel = [v_theta, v_bias]

# Feedforward Pass
cost = T.mean((T.dot(theta, X.T) + bias - y)**2)/2

cost_f = theano.function(inputs=[X, y], outputs=cost,
						 allow_input_downcast=True)

# Backward Pass
gradients = T.grad(cost, params)

def momentum_update(param, velocity, df):
	v_prev = velocity
	v = mu * velocity - lr * df
	x = mu * v_prev + (1-mu) * v
	updates = (param, param+x), (velocity, v)

	return updates

updates = []
for p, v, g in zip(params, vel, gradients):
	param_updates, vel_updates = momentum_update(p, v, g)
	updates.append(param_updates)
	updates.append(vel_updates)

# Theano functions
train = theano.function(inputs=[X, y, mu, lr], outputs=cost,
						updates=updates,
						allow_input_downcast=True)

val_cost = theano.function(inputs=[X, y], outputs=cost,
						   allow_input_downcast=True)


# Main model 
def train_model(training_data, validation_data=None, learning_rate=0.00001,
				velocity=0.9, momentum=0.95, epochs=100, mini_batch_size=20, 
				patience=3):

	assert momentum < 1 and momentum >= 0
	assert velocity < 1 and velocity >= 0
	
	total_values = len(training_data)
	# Best results
	best_loss_train, best_loss_val = np.inf, np.inf
	best_weights_train, best_weights_val = None, None
	best_bias_train, best_bias_val = None, None
	iteration_train, iteration_val = None, None
	# Average validation loss, used for early stoppign
	val_loss_list = np.zeros(patience)

	## FOR TESTING ##
	validation_losses = []

	for epoch in range(epochs):
		np.random.shuffle(training_data)
		mini_batches = [training_data[k: k+mini_batch_size]
						for k in range(0, total_values, mini_batch_size)]

		for mini_batch in mini_batches:
			inputs = mini_batch[:, :-1]
			outputs = mini_batch[:, -1] # this gives it shape (n, ) instead of (n, 1)

			train_loss = train(inputs, outputs, velocity, learning_rate)

		if np.any(validation_data):
			val_input = validation_data[:, :-1]
			val_output = validation_data[:, -1]

			val_loss = val_cost(val_input, val_output)
			## FOR TESTING ##
			validation_losses.append(val_loss)

			if val_loss < best_loss_val:
				best_loss_val = val_loss
				best_weights_val = theta
				best_bias_val = bias
				iteration_val = epoch

			val_loss_list[:-1] = val_loss_list[1:]
			val_loss_list[-1] = val_loss

			if (patience > 0 and
				epoch > patience and
				(np.mean(val_loss_list)/best_loss_val)-1 < 0.01):
				# print('breaking on epoch {}'.format(epoch))
				# break
				print('epoch: {}, loss: {}'.format(epoch, val_loss))
				print(best_weights_val.get_value(), best_bias_val.get_value())
				print('---------------')
				# break

		if train_loss < best_loss_train:
			best_loss_train = train_loss
			best_weights_train = theta
			best_bias_train = bias
			iteration_train = epoch

	print('The best validation result occured on epoch: {}, resulting in a '
		  'loss of {} and yielding weights of {} and a bias of: {}'\
		  .format(iteration_val, best_loss_val, 
				  best_weights_val.get_value(), best_bias_val.get_value()))
	print('==================================================================')
	# print('The best training result occured on epoch: {}, resulting in a '
	# 	  'loss of {} and yielding weights of {} and a bias of: {}'\
	# 	  .format(iteration_train, best_loss_train, 
	# 			  best_weights_train.get_value(), best_bias_train.get_value()))

	print(val_loss_list)
	print(np.mean(val_loss_list))
	# return (best_weights_val, best_bias_val)
	## FOR TESTING ##
	return validation_losses

########################################################################

location_train = 'C:\\Users\\lukas\\OneDrive\\Documents\\Machine Learning' +\
				 '\\Theano\\Linear_Regeression\Data\\data_train.npy'
location_valid = 'C:\\Users\\lukas\\OneDrive\\Documents\\Machine Learning' +\
				 '\\Theano\\Linear_Regeression\Data\\data_val.npy'

training_data = np.load(location_train)
validation_data = np.load(location_valid)
y = train_model(training_data, validation_data)

# m = 20
# x = np.arange(len(y[m:]))

# plt.figure(figsize=(40,20))
# plt.plot(x, y[m:])
# plt.show()

# Ensemble technique
# weights_ens, bias_ens = 0, 0
# n = 10
# for i in range(n):
# 	weights, bias = train_model(training_data, validation_data)

# 	weights_ens += weights.get_value()
# 	bias_ens += bias.get_value()

# print('Ensemble weights: {}, ensemble bias: {}'
# 	  .format(weights_ens/n, bias_ens/n))

########################################################################

