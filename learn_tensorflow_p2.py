import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.mul(x1,x2)

print(result)

# sess = tf.Session()
# print(sess.run(result))
# sess.close()

with tf.Session() as sess:
		output = sess.run(result)
		print (output)

print(output)

'''
input > weights > hidden layer 1 (activation function)
> weights > hidden layer 2 (activation function) >
weights > output layer 

compare output to intended output > cost or loss function 
(cross entropy) optimization function (optimizer) 
> minimize cost (AdamOptimizer ... SGD)

feed forward + backprop = epoch

'''





