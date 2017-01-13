import tensorflow as tf 

'''
Replace the separate nodes a and b with a 
consolidated input node

tf.reduce_prod() and tf.reduce_sum(). These functions
, when just given a Tensor as input, take all of
its values and either multiply or sum them up, 
respectively.
'''
a = tf.constant([5,3], name="input_a")
b = tf.reduce_prod(a, name="prod_b")
c = tf.reduce_sum(a, name="sum_c")
d = tf.add(b,c, name="add_d")

with tf.Session() as sess:
	print(sess.run(d))

	writer = tf.summary.FileWriter("./my_graph1", sess.graph)

	# remember to close session. It is a good practice
	sess.close()

