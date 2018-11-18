import tensorflow as tf
import numpy as np
import time
import Dataset_bmi as Dataset
import scipy
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
# on tensorflow
start_time = time.time()
img_number = 4  # 5
batch_size = 64
max_epoch = 500
learning_rate = 0.0001  # 0.0001
img_size = 227
# placeholder
img_0 = tf.placeholder(tf.float32, shape=[None, 227, 227, 3])
label = tf.placeholder(tf.float32, shape=[1, None])
keep_prob = tf.placeholder(tf.float32)
# load the pretrained Alexnet
net_data = np.load("bvlc_alexnet.npy",encoding='latin1').item()

regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)


def conv(inputt, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
	c_i = inputt.get_shape()[-1]
	assert c_i % group == 0
	assert c_o % group == 0
	convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
	if group == 1:
		conv = convolve(inputt, kernel)
	else:
		input_groups = tf.split(inputt, group, 3)
		kernel_groups = tf.split(kernel, group, 3)
		output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
		conv = tf.concat(output_groups, 3)
	return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])


def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):  # use the xavier initializer
	# initial = tf.truncated_normal(shape, stddev = 0.01)
	return tf.get_variable('weight', shape=shape, initializer=tf.contrib.layers.xavier_initializer(),
	                       regularizer=regularizer)


def bias_variable(shape):
	initial = tf.constant(0.01, shape=shape)
	return tf.get_variable('bias', initializer=initial, regularizer=regularizer)


# the encoder of images, takes in a list of images, transformations, and flows
def encoder(img_l):
	nodes_list = []
	with tf.variable_scope('encoder_img') as en_img:
		count = 0
		for i in range(len(img_l)):
			count = count + 1
			if count == 2:
				en_img.reuse_variables()
			with tf.variable_scope('img_layer1'):
				W_conv1 = tf.get_variable('weight', initializer=net_data["conv1"][0], regularizer=regularizer)
				b_conv1 = tf.get_variable('bias', initializer=net_data["conv1"][1], regularizer=regularizer)
				h_conv1 = tf.nn.relu(tf.nn.conv2d(img_l[i], W_conv1, strides=[1, 4, 4, 1], padding='VALID') + b_conv1)
				radius = 2;
				alpha = 2e-05;
				beta = 0.75;
				bias = 1.0
				lrn1 = tf.nn.local_response_normalization(h_conv1, depth_radius=radius, alpha=alpha, beta=beta,
				                                          bias=bias)
				h_pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
			with tf.variable_scope('img_layer2'):
				k_h = 5;
				k_w = 5;
				c_o = 256;
				s_h = 1;
				s_w = 1;
				group = 2
				W_conv2 = tf.get_variable('weight', initializer=net_data["conv2"][0], regularizer=regularizer)
				b_conv2 = tf.get_variable('bias', initializer=net_data["conv2"][1], regularizer=regularizer)
				# h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides = [1,1,1,1], padding='SAME') + b_conv2)
				conv2_in = conv(h_pool1, W_conv2, b_conv2, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
				h_conv2 = tf.nn.relu(conv2_in)
				radius = 2;
				alpha = 2e-05;
				beta = 0.75;
				bias = 1.0
				lrn2 = tf.nn.local_response_normalization(h_conv2, depth_radius=radius, alpha=alpha, beta=beta,
				                                          bias=bias)
				h_pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
			with tf.variable_scope('img_layer3'):
				W_conv3 = tf.get_variable('weight', initializer=net_data["conv3"][0], regularizer=regularizer)
				b_conv3 = tf.get_variable('bias', initializer=net_data["conv3"][1], regularizer=regularizer)
				h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)
			with tf.variable_scope('img_layer4'):
				k_h = 3;
				k_w = 3;
				c_o = 384;
				s_h = 1;
				s_w = 1;
				group = 2
				W_conv4 = tf.get_variable('weight', initializer=net_data["conv4"][0], regularizer=regularizer)
				b_conv4 = tf.get_variable('bias', initializer=net_data["conv4"][1], regularizer=regularizer)
				# h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, strides = [1,1,1,1], padding='SAME') + b_conv4)
				conv4_in = conv(h_conv3, W_conv4, b_conv4, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
				h_conv4 = tf.nn.relu(conv4_in)
			with tf.variable_scope('img_layer5'):
				k_h = 3;
				k_w = 3;
				c_o = 256;
				s_h = 1;
				s_w = 1;
				group = 2
				W_conv5 = tf.get_variable('weight', initializer=net_data["conv5"][0], regularizer=regularizer)
				b_conv5 = tf.get_variable('bias', initializer=net_data["conv5"][1], regularizer=regularizer)
				# h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5, strides = [1,1,1,1], padding='SAME') + b_conv5)
				conv5_in = conv(h_conv4, W_conv5, b_conv5, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
				h_conv5 = tf.nn.relu(conv5_in)
				h_pool5 = tf.nn.max_pool(h_conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
			# fc1, to 4096
			with tf.variable_scope('img_fc1'):
				W_fc1 = tf.get_variable('weight', initializer=net_data["fc6"][0], regularizer=regularizer)
				b_fc1 = tf.get_variable('bias', initializer=net_data["fc6"][1], regularizer=regularizer)
				# h_pool5_flat = tf.reshape(h_pool5, [-1, 13 * 13 * 256])
				# h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)
				h_fc1 = tf.nn.relu_layer(tf.reshape(h_pool5, [-1, int(np.prod(h_pool5.get_shape()[1:]))]), W_fc1, b_fc1)
				# h_fc1_out = tf.nn.dropout(h_fc1, keep_prob)
			with tf.variable_scope('img_fc2'):
				W_fc2 = tf.get_variable('weight', initializer=net_data["fc7"][0], regularizer=regularizer)
				b_fc2 = tf.get_variable('bias', initializer=net_data["fc7"][1], regularizer=regularizer)
				h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
				# h_fc2_out = tf.nn.dropout(h_fc2, keep_prob)
			##append the nodes_list
			nodes_list.append(h_fc2)
	#     tf.summary.histogram('b_conv6', b_conv6)
	return nodes_list


def read_out(n_list):
	with tf.variable_scope('read_out') as read_out:
		out_list = []
		count = 0
		for i in range(len(n_list)):
			if count == 1:
				read_out.reuse_variables()
			W_out = weight_variable([4096, 1])
			b_out = bias_variable([1])
			h_out = tf.reshape(tf.matmul(n_list[i], W_out) + b_out, [-1])
			out_list.append(h_out)
			count = count + 1
		return out_list


def run_test(sess, dataset, epoch):
	test_batch_size = 1000
	test_batch = dataset.test_batch(test_batch_size, 1)
	test_mean_distance = sess.run(mean_distance, feed_dict={img_0: test_batch[0][0], label: test_batch[1],
	                                                        keep_prob: 1}) / test_batch_size
	print(">>>>>>>>>>>>>>>>>>>>>>>>test_mean_distance:%f" % test_mean_distance)
	fp = open('log_bmi_02_26.txt', 'a')
	fp.write("epoch:%d, test_mean_distance:%f \n\n" % (epoch, test_mean_distance))
	fp.close()


#  for m in range(2):
#    if m==0:
#      test_batch = dataset.test_batch(220)
#      tag = 'test'
#    else:
#      test_batch = dataset.test_trainset(220)
#      tag = 'train'


def run_train(sess, train_step, datasetname, attributes_num):
	dataset = Dataset.reader()
	epoch = 0
	step = 0
	fp = open('log_relative_attributes_02_20_2.txt', 'a')
	fp.write('Start training once........................\n')
	fp.write('attributes:' + str(attributes_num) + '\n')
	fp.close()
	run_test(sess, dataset, epoch)  #### test

	while (epoch <= max_epoch):

		next_batch, epoch_end = dataset.next_batch(batch_size, 1)
		train_step.run(feed_dict={img_0: next_batch[0][0], label: next_batch[1], keep_prob: 0.5})
		if step % 100 == 0:
			print("epoch: %d, step: %d, loss: %f" % (
			epoch, step, sess.run(loss, feed_dict={img_0: next_batch[0][0], label: next_batch[1], keep_prob: 1})))

			#      print "batch_label",next_batch[0]
			print("mean_distance:%f" % (sess.run(mean_distance, feed_dict={img_0: next_batch[0][0], label: next_batch[1],
			                                                         keep_prob: 1}) / batch_size))

			print("reg_term:%f" % (
				sess.run(reg_term, feed_dict={img_0: next_batch[0][0], label: next_batch[1], keep_prob: 1})))


		step += 1
		if epoch_end == 1:
			if epoch % 2 == 0:
				run_test(sess, dataset, epoch)
				saver.save(sess, './poison_model/model.ckpt')
			epoch = epoch + 1
			step = 0



###construct the network###
with tf.device('/gpu:' + args.gpu):
	img_list = [img_0]
	# use list here just because the code is adapted from my original GNN construction.
	nodes_list = encoder(img_list)
	final_output_list = read_out(nodes_list)

	# calculate loss
	loss = 0
	mean_distance = 0
	for i in range(len(nodes_list)):
		P = final_output_list[i]
		L = label[i, :]
		loss = loss + tf.nn.l2_loss(P - L)
		mean_distance = mean_distance + tf.reduce_sum(tf.abs(P - L))
	mean_distance = mean_distance / len(nodes_list)
	# add regularization
	reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
	#loss = loss + reg_term
	# train step and configuration
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	saver = tf.train.Saver()
	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth = True


### main func ###
with tf.Session(config=config) as sess:
	sess.run(tf.global_variables_initializer())
### use this saver.restore to restore the model
	saver.restore(sess, './poison_model/model.ckpt')
	run_train(sess, train_step, 'OSR', i)

	# summary writer
#  if args.summary == 1:
#    writer = tf.summary.FileWriter("./tmp/relative-mpnn")
#    summaries = tf.summary.merge_all()

end_time = time.time()
print('training complete')
