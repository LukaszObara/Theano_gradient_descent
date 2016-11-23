"""Tutorial to implement (S)GD on linear regression using basic python
code and theano.py"""

#### Libraries
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
					  value=np.random.uniform(-1.0, 1.0, size=(3))
					  	.astype(theano.config.floatX))
bias = theano.shared(name='bias',
					 value=np.random.uniform(0, 20.0, size=(1, 1))
						.astype(theano.config.floatX),
					 broadcastable=(False, True))

params = [theta, bias]

# Feedforward Pass
cost = T.mean((T.dot(theta, X.T) + bias - y)**2)/2

cost_f = theano.function(inputs=[X, y], outputs=cost,
						 allow_input_downcast=True)

# Backward Pass
gradients = T.grad(cost, params)
updates = [(p, p-lr*g) for p, g in zip(params, gradients)]

# Theano functions
train = theano.function(inputs=[X, y, lr], outputs=cost,
						updates=updates,
						allow_input_downcast=True)

val_cost = theano.function(inputs=[X, y], outputs=cost,
						   allow_input_downcast=True)

# Main model 
def train_model(training_data, validation_data=None, learning_rate= 0.0001,
				epochs=100, mini_batch_size=20, patience=3, tolerance=0.05,
				display_results=True):

	total_values = len(training_data)
	# Best results
	best_loss_train, best_loss_val = np.inf, np.inf
	best_weights_train, best_weights_val = None, None
	best_bias_train, best_bias_val = None, None
	iteration_train, iteration_val = None, None
	# Average validation loss, used for early stoppign
	val_loss_list = np.zeros(patience)

	training_losses, validation_losses = [], []

	for epoch in range(epochs):
		np.random.shuffle(training_data)
		mini_batches = [training_data[k: k+mini_batch_size]
						for k in range(0, total_values, mini_batch_size)]

		av_loss = 0 

		for mini_batch in mini_batches:
			inputs = mini_batch[:, :-1]
			outputs = mini_batch[:, -1] # this gives it shape (n, ) instead of (n, 1)

			train_loss = train(inputs, outputs, learning_rate)
			av_loss += train_loss/len(mini_batch)

		training_losses.append(av_loss)

		if np.any(validation_data):
			val_input = validation_data[:, :-1]
			val_output = validation_data[:, -1]

			val_loss = val_cost(val_input, val_output)
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
				(np.mean(val_loss_list)/best_loss_val)-1 < tolerance):
				
				if display_results:
					print('Breaking on epoch: {}, returning a loss: {}'
						  .format(epoch, val_loss))
					print(best_weights_val.get_value(), best_bias_val.get_value())
					print('---------------')
				break

		if av_loss < best_loss_train:
			best_loss_train = av_loss
			best_weights_train = theta
			best_bias_train = bias
			iteration_train = epoch

	if display_results:
		print('The best validation result occured on epoch: {}, resulting in a'
			  ' loss of {} and yielding weights of {} and a bias of: {}'\
			  .format(iteration_val, best_loss_val, 
					  best_weights_val.get_value(), best_bias_val.get_value()))
		print('==============================================================')
		print('The best training result occured on epoch: {}, resulting in a '
			  'loss of {} and yielding weights of {} and a bias of: {}'\
			  .format(iteration_train, best_loss_train, 
					  best_weights_train.get_value(), 
					  best_bias_train.get_value()))

	return training_losses, validation_losses, epoch

###############################################################################

location_train = 'C:\\Users\\lukas\\OneDrive\\Documents\\Machine Learning' +\
				 '\\Theano\\Linear_Regeression\Data\\data_train.npy'
location_valid = 'C:\\Users\\lukas\\OneDrive\\Documents\\Machine Learning' +\
				 '\\Theano\\Linear_Regeression\Data\\data_val.npy'

training_data = np.load(location_train)
validation_data = np.load(location_valid)
t_loss, v_loss, _ = train_model(training_data, validation_data, 
								display_results=True)

m = 5
x = np.arange(len(t_loss[m:]))

fig = plt.figure(figsize=(40, 20))
ax = fig.add_subplot(1, 1, 1) 
                                   
major_ticks_x = np.arange(0, 40, 10)                                              
minor_ticks_x = np.arange(0, 40, 5)
major_ticks_y = np.arange(0, 450, 50)                                              
minor_ticks_y = np.arange(0, 450, 10)

ax.set_xticks(major_ticks_x)                                                       
ax.set_xticks(minor_ticks_x, minor=True)                                           
ax.set_yticks(major_ticks_y)                                                       
ax.set_yticks(minor_ticks_y, minor=True) 

ax.grid(which='both')
ax.grid(which='minor', alpha=0.7)                                                
ax.grid(which='major', alpha=1.0)    

t_plot = plt.plot(x, t_loss[m:], label='Training Loss')
v_plot = plt.plot(x, v_loss[m:], label='Validation Loss')                                            

plt.legend(loc='upper right')
# plt.show()

# Using an Ensemble
# weights_ens, bias_ens, epochs = 0, 0, 0
# n = 50
# for i in range(n):
# 	_, _, epoch = train_model(training_data, validation_data,
# 							  display_results=False)

# 	epochs += epoch

# 	weights_ens += theta.get_value()
# 	bias_ens += bias.get_value()

# 	theta_reset = np.random.uniform(-1.0, 1.0, size=(3))\
# 					.astype(theano.config.floatX)
# 	bias_reset = np.random.uniform(0, 20.0, size=(1, 1))\
# 						.astype(theano.config.floatX)
# 	theta.set_value(theta_reset)
# 	bias.set_value(bias_reset)

# print('Ensemble weights: {}, ensemble bias: {}, average number of epochs: {}'
# 	  .format(weights_ens/n, bias_ens/n, epochs/n))

########################################################################

