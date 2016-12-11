import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data = tf.random_normal([2,20])


a=tf.constant(5,name="input_a")
b=tf.constant(3,name="input_b")
c=tf.mul(a,b,name="mul_c")
d=tf.add(a,b,name="add_d")
e=tf.add(c,d,name="add_e")

with tf.Session() as sess:
	print(sess.run(e))
	out = sess.run(data)
	x, y = out

	plt.scatter(x, y)
	plt.show()




