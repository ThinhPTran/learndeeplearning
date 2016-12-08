# import tensorflow as tf
#
# # x = 35
# # y = x + 5
# # print(y)
#
# x = tf.constant(35, name='x')
# y = tf.Variable(x + 5, name='y')
#
# model = tf.global_variables_initializer()
#
# with tf.Session() as session:
#     session.run(model)
#     print(session.run(y))


import tensorflow as tf
a=tf.constant(5,name="input_a")
b=tf.constant(3,name="input_b")
c=tf.mul(a,b,name="mul_c")
d=tf.add(a,b,name="add_d")
e=tf.add(c,d,name="add_e")