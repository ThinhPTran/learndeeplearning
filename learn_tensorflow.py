import tensorflow as tf 

'''
x isn't a specific value. it's a placeholder, a value that we'll input
when we ask TensorFlow to run a computation. We want to be able to 
input any number of MNIST images, each flattened into a 784-dimensional
vector. We represent this as a 2-D tensor of floating-point numbers,
with a shape [None, 784]. (Here None means that a dimension can be
of any length.)
'''

x = tf.placeholder(tf.float32, [None, 784])

'''
We slso need the weights and biases for our model. We could 
imagine treating these like additional inputs
'''

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

'''
We can now implement our model
'''
y = tf.nn.softmax(tf.matmul(x, W) + b)

'''
In Tensorflow. when the model is defined, it can be run on any 
devices such sas CPU, GPU or even phones!
'''

'''
In order to train our model, we need to define what it means
for the model to be good. Well, actually, in machine learning 
we typically define what it means for a model to be bad. We 
call this the cost, or the loss and it represents how far off
outr model is from our desired outcome. We try to minimize that
error, and the smaller the error margin, the better our model is.

One very common, very nice function to determine the loss of a
model called "cross-entropy". Cross-entropy arises from thinking
about information compressing codes in information theory but
it winds up being an important idea in lots of areas, from 
gambling to machine learning. 

To implement cross-entropy we nee to first add a new placeholder 
to input the correct answers
'''

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

'''
(Note that in the source code, we don't use this formulation, 
because it is numerically unstable. Instead, we apply 
tf.nn.softmax_cross_entropy_with_logits on the unnormalized logits 
(e.g., we call softmax_cross_entropy_with_logits on tf.matmul(x, W) + b),
 because this more numerically stable function internally computes
  the softmax activation. In your code, 
  consider using tf.nn.(sparse_)softmax_cross_entropy_with_logits instead).
'''

'''
Now that we know what we want our model to do,
it's very easy to have TensorFlow train it to do so.
Because TensorFlow knows the entire graph of your computations,
it can automatically use the backpropagation algorithm to 
efficiently determine how your variables affect the loss 
you ask it to minimize. Then it can apply your choice
of optimization algorithm to modify the variables and
reduce the loss.
'''

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

'''
In this case, we ask TensorFlow to minimize cross_entropy 
using the gradient descent algorithm with a learning rate of 0.5.
Gradient descent is a simple procedure, where TensorFlow simply
shifts each variable a little bit in the direction that reduces
the cost. But TensorFlow also provides many other optimization
algorithms: using one is as simple as tweaking one line.
'''

'''
Now we have our model set up to train. One last thing
before we launch it, we have to create an operation to 
initialize the variables we created. Note that this defines
the operation but does not run it yet:
'''

init = tf.global_variables_initializer()

'''
We can now launch the model in a Session, and now we run 
the operation that initializes the variables
'''

sess = tf.Session()
sess.run(init)

'''
Let'train -- we'll run the training step 1000 times!
'''

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

