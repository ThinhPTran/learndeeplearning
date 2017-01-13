import tensorflow as tf 
import numpy as np 

'''
tf.placeholder takes in a required parameter
dtype, as well as the optional parameter shape.
+	dtype specifies the data type of values
that will be passed into the placeholder. This
is required, as it is needed to ensure that 
there will be no type mismatch errors. 
+	specifies what shape the fed Tensor will be.
'''

# Creates a placeholder vector of length 2 
# with data type int32
a = tf.placeholder(tf.int32, shape=[2], name="my_input")

# use the placeholder as if it were any other
# Tensor object
b = tf.reduce_prod(a, name="prod_b") 
c = tf.reduce_sum(a, name="sum_c")

# Finish off the graph
d = tf.add(b, c, name="add_d")

with tf.Session() as sess:

	# Create a dictionary to pass into feed_dict
	# Key: 'a', the handle to the placeholder's
	# output Tensor. 
	# Value: A vector with value [5, 3] and int32
	# datatype  
	input_dict = {a: np.array([5,3], dtype=np.int32)}

	# Fetch the value of 'd', feeding the value
	# of 'input_vector' into 'a'
	print(sess.run(d, feed_dict=input_dict))

	writer = tf.summary.FileWriter("./my_graph4", sess.graph)
	writer.close()

	sess.close()





