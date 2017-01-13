import tensorflow as tf 



# Create Operations, Tensors, etc (using the default graph)
a = tf.add(2,5)
b = tf.mul(a, 3)

# Start up a "Session" using the default graph
'''
Note that these two calls are identical
sess = tf.Session()
sess = tf.Session(graph=tf.get_default_graph())
'''

with tf.Session() as sess: 
	'''
	Once a Session is opened, you can use its primary
	method, run(), to calculate the value of a desired
	Tensor output:
	'''
	sess.run(b)

	'''
	Session.run() takes in one required parameter,
	fetches, as well as three optional parameters:
	feed_dict, options and run_metadata. 

	fetches accepts any graph element (either an Operation
 	or Tensor object), which specifies what the user
 	would like to execute. If the requested object
 	is a Tensor, then the output of run() will be
 	a NumPy array. If the object is an Operation, then
 	the output will be None. 

	In the above example, we set fetches to the tensor
	b (the output of the tf.mul Operation). This tells
	Tensorflow that the Session should find all of 
	the nodes necessary to compute the value of b,
	execute them in order, and output the value of b 
	'''

	'''
	We can also pass in a list of graph elements:
	'''

	sess.run([a,b]) # return [7, 21]

	writer = tf.summary.FileWriter("./my_graph2",sess.graph)
	writer.close()

	sess.close()
