import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 


a = tf.constant(5, name="input_a")
b = tf.constant(3, name="input_b")
c = tf.mul(a,b, name="mul_c")
d = tf.add(a,b, name="add_d")
e = tf.add(c,d, name="sum_e")
e1 = tf.add(e,b, name="sum_e1")

with tf.Session() as sess:
	out = sess.run(e1)
	print(out)
	

writer = tf.train.SummaryWriter('./my_graph', sess.graph)
writer.close(); 


