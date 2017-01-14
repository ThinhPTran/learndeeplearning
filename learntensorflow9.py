# import tensorflow library
import tensorflow as tf 

# create a new graph
graph = tf.Graph()


# set it as default graph
with graph.as_default():

	with tf.name_scope("variables"):
		# Variable to keep track of how many times the graph has been run
		global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")

		# Variable that keeps track of the sum of all output values over time:
		total_output = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="total_output")


	with tf.name_scope("transformation"):

		# Separate input layer
		with tf.name_scope("input"):
			# Create input placeholder - takes in a vector
			a = tf.placeholder(tf.float32, shape=[None], name="input_placeholder_a")

		# Separate middle layer
		with tf.name_scope("intermediate_layer"):
			b = tf.reduce_prod(a, name="product_b")
			c = tf.reduce_sum(a, name="sum_c")

		# Separate output layer
		with tf.name_scope("output"):
			output = tf.add(b, c, name="output")

	with tf.name_scope("update"):
		# Increments the total_output variable by the latest output
		update_total = total_output.assign_add(output)

		# Increments the above 'global_step' variable, should be run whenever the graph is run
		increment_step = global_step.assign_add(1)

	with tf.name_scope("summaries"):
		avg = tf.div(update_total, tf.cast(increment_step, tf.float32), name="average")

		# print("output", output)

		# Creates summaries for output node
		tf.summary.scalar(b'Output', output)
		tf.summary.scalar(b'Sum of outputs over time', update_total)
		tf.summary.scalar(b'Average of outputs over time', avg)

	with tf.name_scope("global_ops"):
		# Initialization Op
		init = tf.global_variables_initializer()

		# Merge all summaries into one Operation
		merged_summaries = tf.summary.merge_all()

def run_graph(sess, writer, input_tensor):
	feed_dict = {a: input_tensor}
	_, step, summary = sess.run([output, increment_step, merged_summaries], feed_dict=feed_dict)
	# writer.add_summary(summary, global_step=step)

with tf.Session(graph=graph) as sess: 

	writer = tf.summary.FileWriter('./improved_graph', graph)
	sess.run(init)

	run_graph(sess, writer, [2,8])
	run_graph(sess, writer, [3,1,3,3])
	run_graph(sess, writer, [8])
	run_graph(sess, writer, [1,2,3])
	run_graph(sess, writer, [11,4])
	run_graph(sess, writer, [4,1])
	run_graph(sess, writer, [7,3,1])
	run_graph(sess, writer, [6,3])
	run_graph(sess, writer, [8,2])
	run_graph(sess, writer, [4,5,6])

	# Write the summaries to disk
	writer.flush()

	# Close the summaryWriter
	writer.close() 

	# Close the session
	sess.close()
