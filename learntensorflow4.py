import tensorflow as tf 

# Create Operations, Tensors, etc (using the default graph)
a = tf.add(2, 5)
b = tf.mul(a, 3)

# Start up a "Session" using the default graph
sess = tf.Session()

# Define a dictionary that says to replace the 
# default value of a with 15
replace_dict = {a: 15}

print(sess.run(b, feed_dict=replace_dict))