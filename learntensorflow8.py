import tensorflow as tf 

with tf.name_scope("Scope_A"):
	a = tf.add(1, 2, name="A_add")
	b = tf.mul(a, 3, name="A_mul")

with tf.name_scope("Scope_B"):
	c = tf.add(4, 5, name="B_add")
	d = tf.mul(c, 6, name="B_mul")

e = tf.add(b, d, name="output")

writer = tf.train.SummaryWriter('./name_scope_1', graph=tf.get_default_graph())
writer.close()