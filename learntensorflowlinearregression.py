import tensorflow as tf 

# initialize variables/model parameters
W = tf.Variable(tf.zeros([2, 1]), name="weights")
b = tf.Variable(0., name="bias")

# define the training loop operations
def inference(X):
	# compute inference model over data X and return the result
	return tf.matmul(X, W) + b 

def loss(X, Y):
	# compute loss over training data X and expected outputs Y
	Y_predicted = inference(X)
	return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))

def inputs():
	# read/generate input training data X and expected outputs Y
	weight_age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25], [63, 28], [72, 36], [79, 57], [75, 44], [27, 24], [89, 31], [65, 52], [57, 23], [59, 60], [69, 48], [60, 34], [79, 51], [75, 50], [82, 34], [59, 46], [67, 23], [85, 37], [55, 40], [63, 30]]
	blood_fat_content = [354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209, 290, 346, 254, 395, 434, 220, 374, 308, 220, 311, 181, 274, 303, 244]

	return tf.to_float(weight_age), tf.to_float(blood_fat_content)

def train(total_loss):
	# train / adjust model parameters according to computed total loss
	learning_rate = 0.0000001
	return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

def evaluate(sess, X, Y):
	# evaluate the resulting trained model
	print sess.run(inference([[80., 25.]])) # 303
	print sess.run(inference([[65., 25.]])) # 256

tf.summary.scalar(b'bias', b)

init = tf.global_variables_initializer()
merged_summaries = tf.summary.merge_all()

# Launch the graph in a session, setup boilerplate
with tf.Session() as sess:

	# tf.initialize_all_variables().run()
	sess.run(init)

	X, Y = inputs()

	total_loss = loss(X, Y)
	train_op = train(total_loss)

	# coord = tf.train.Coordinator()
	# threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	writer = tf.summary.FileWriter('./testgraph', sess.graph)

	# actual training loop 
	training_steps = 1000
	for step in range(training_steps):
		_, summary = sess.run([train_op, merged_summaries])
		writer.add_summary(summary, global_step=step)

		# for debugging  and learning purposes, see how the loss gets decremented through train steps
		if step % 100 == 0:
			print "loss: ", sess.run(total_loss)
			print W.eval()
			print b.eval()


	evaluate(sess, X, Y)


	writer.flush()
	writer.close()

	# coord.request_stop()
	# coord.join(threads)
	sess.close(); 




