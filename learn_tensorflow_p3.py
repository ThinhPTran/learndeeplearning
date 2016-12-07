
'''
input > weights > hidden layer 1 (activation function)
> weights > hidden layer 2 (activation function) >
weights > output layer 

compare output to intended output > cost or loss function 
(cross entropy) optimization function (optimizer) 
> minimize cost (AdamOptimizer ... SGD)

feed forward + backprop = epoch

'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
The MNIST data is split into three parts: 55,000 data points
 of training data (mnist.train), 10,000 points of test data 
 (mnist.test), and 5,000 points of validation data 
 (mnist.validation). This split is very important: 
 it's essential in machine learning that we have separate data 
 which we don't learn from so that we can make sure that 
 what we've learned actually generalizes!
'''

mnist = input_data.read_data_sets("/home/thinhptran/Programming/LearnDeepLearning", one_hot=True)

# 10 classes, 0-9
'''
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
'''

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

# height x width: 784 = 28x28
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


#This is computational graph
def neural_network_model(data):

	# (inputdata * weights) + biases  
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1]))
	               ,'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2]))
	               ,'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
	               
	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3]))
	               ,'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes]))
	               ,'biases':tf.Variable(tf.random_normal([n_classes]))}	                               
 
	# (input_data * weights) + biases

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']) 
		,hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)
	
	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']) 
		,hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']) 
		,hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

	return output


#specify how data runs through the model
def train_neural_network(x):
	prediction = neural_network_model(x)

	'''
	cost function is used in below form for the numerical stability purposes. 
	'''
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y))

	#learning rate = 0.001 
	# Optimizer can be call train_step 
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	# cycles  of feedforward + backprop
	hm_epochs = 10

	with tf.Session() as sess:
		'''
		Now we have our model set up to train. One last thing
		before we launch it, we have to create an operation to 
		initialize the variables we created. Note that this defines
		the operation but does not run it yet:
		'''	
		sess.run(tf.initialize_all_variables())

		for epoch in range(hm_epochs):
			# To measure loss, failure for each epoch
			epoch_loss = 0
			# Loop through training dataset 
			for _ in range(int(mnist.train.num_examples/batch_size)):
				# It is training dataset in a size of batch size
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', hm_epochs,'loss:', epoch_loss)

		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)
