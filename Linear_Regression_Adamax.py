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

rng = np.random.RandomState(35)

# Paramater iniitilization
# Symbolic variable
X = T.matrix(name='X', dtype=theano.config.floatX)
y = T.vector(name='y', dtype=theano.config.floatX)
lr = T.scalar(name='learn_rate', dtype=theano.config.floatX)

# Variables that will be updated, hence are declared as `theano.share`
theta = theano.shared(name='theta', 
					  value=rng.uniform(-1.0, 1.0, size=(3))
					  	.astype(theano.config.floatX))
bias = theano.shared(name='bias',
					 value=rng.uniform(0, 20, size=(1, 1))
						.astype(theano.config.floatX),
					 broadcastable=(True, True))

# ADAM Parameters
beta1 = T.scalar(name='beta1', dtype=theano.config.floatX)
beta2 = T.scalar(name='beta2', dtype=theano.config.floatX)
t = theano.shared(name='iteration', value=np.float32(1.0))
m_theta = theano.shared(name='moment_theta',
						value=np.zeros(3, dtype=theano.config.floatX))
m_bias = theano.shared(name='moment_bias',
					   value=np.zeros((1,1), dtype=theano.config.floatX),
					   broadcastable=(True, True))
u_theta = theano.shared(name='up_theta',
						value=np.zeros(3, dtype=theano.config.floatX))
u_bias = theano.shared(name='up_bias',
					   value=np.zeros((1, 1), dtype=theano.config.floatX),
					   broadcastable=(True, True))

params = [theta, bias]
moments = [m_theta, m_bias]
upd = [u_theta, u_bias]

one = T.constant(1.0)

# Feedforward Pass
cost = T.mean((T.dot(theta, X.T) + bias - y)**2)/2

cost_f = theano.function(inputs=[X, y], outputs=cost,
							  allow_input_downcast=True)

# Backward Pass
gradients = T.grad(cost, params)

grads = theano.function(inputs=[X, y], outputs=gradients,
						allow_input_downcast=True)

def update_rule(param, moment, u, df):
	m_t = beta1 * moment + (one-beta1) * df
	u_t = T.maximum(beta2*u, T.abs_(df))
	x = (lr/(1-beta1**t)) * (m_t/u_t) 
	updates = (param, param-x), (moment, m_t), (u, u_t)

	return updates

updates = []
for p, m, u, g in zip(params, moments, upd, gradients):
	p_update, m_update, u_update = update_rule(p, m, u, g)
	updates.append(p_update)
	updates.append(m_update)
	updates.append(u_update)
updates.append((t, t+1))

# Theano function
train = theano.function(inputs=[X, y, lr, beta1, beta2], outputs=cost,
						updates=updates,
						allow_input_downcast=True)

# Training model
def train_model(training_data, validation_data=None, learning_rate=1e-0, 
				beta_1=0.9, beta_2=0.999, epochs=100, mini_batch_size=10,
				patience=5):

	total_values = len(training_data)
	# Performance monitoring
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
			outputs = mini_batch[:, -1]

			train_loss = train(inputs, outputs, learning_rate, beta_1, beta_2) 

		if np.any(validation_data):
			val_input = validation_data[:, :-1]
			val_output = validation_data[:, -1]

			val_loss = cost_f(val_input, val_output)
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
				# print('breaking on epoch {}'.format(i))
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
	print('The best training result occured on epoch: {}, resulting in a '
		  'loss of {} and yielding weights of {} and a bias of: {}'\
		  .format(iteration_train, best_loss_train, 
				  best_weights_train.get_value(), best_bias_train.get_value()))

	## FOR TESTING ##
	return validation_losses
########################################################################

location_train = 'C:\\...\\Theano\\Linear_Regeression\Data\\data_train.npy'
location_valid = 'C:\\...\\Theano\\Linear_Regeression\Data\\data_val.npy'

training_data = np.load(location_train)
validation_data = np.load(location_valid)
y = train_model(training_data, validation_data, epochs=100)
print(theta.get_value())
