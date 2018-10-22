import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import data_preprocess as dp

layers = tf.contrib.layers

# CNN hidden layers configuration
CONV_LEN = 3
CONV_LEN_INTE = 3#4
CONV_LEN_LAST = 3#5
CONV_NUM = 64
CONV_MERGE_LEN = 8
CONV_MERGE_LEN2 = 6
CONV_MERGE_LEN3 = 4
CONV_NUM2 = 64
INTER_DIM = 120
CONV_KEEP_PROB = 0.8

# Data sample configuration
FEATURE_DIM = 4 # Total dimension of multimodal inputs
BATCH_SIZE = 64 # Batch size of training data, also the stride for preprocessing
TOTAL_ITER_NUM = 10000 #1000000000
OUT_DIM = 2#len(idDict)

# Training/evaluation configuration
TRAIN_SIZE = 3200 # Train data size
EVAL_DATA_SIZE = TRAIN_SIZE # Calibrated with train data
EVAL_ITER_NUM = int(math.ceil(EVAL_DATA_SIZE / BATCH_SIZE)) # One batch, one evaluation

# Directory macros
DIR_RAW = 'sensorData/raw/'
DIR_LABELS = 'sensorData/label/'
DIR_PROCESSED = 'sensorData/processed/'
PROCESSED_FILE = 'processed.csv'


# Processed data import method
def read_audio_csv(filename_queue):
	reader = tf.TextLineReader()
	key, value = reader.read(filename_queue)
	defaultVal = [[0.] for idx in range(FEATURE_DIM + OUT_DIM)]
	fileData = tf.decode_csv(value, record_defaults=defaultVal)
	features = fileData[:FEATURE_DIM]
	features = tf.reshape(features, [1, FEATURE_DIM]) # WIDE, FEATURE_DIM
	labels = fileData[FEATURE_DIM:]
	return features, labels

def input_pipeline(filenames, batch_size, shuffle_sample=True, num_epochs=None): # batch_size == stride in preprocessing
	# Queue configuration
	filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle_sample)
	example, label = read_audio_csv(filename_queue)
	min_after_dequeue = 1000#int(0.4*len(csvFileList)) #1000
	capacity = min_after_dequeue + 3 * batch_size
	# Shuffle or not
	if shuffle_sample:
		example_batch, label_batch = tf.train.shuffle_batch(
			[example, label], batch_size=batch_size, num_threads=16, capacity=capacity,
			min_after_dequeue=min_after_dequeue)
	else:
		example_batch, label_batch = tf.train.batch(
			[example, label], batch_size=batch_size, num_threads=16)
	return example_batch, label_batch


def batch_norm_layer(inputs, phase_train, scope=None):
	if phase_train:
		return layers.batch_norm(inputs, is_training=True, scale=True,
			updates_collections=None, scope=scope)
	else:
		return layers.batch_norm(inputs, is_training=False, scale=True,
			updates_collections=None, scope=scope, reuse = True)

# DeepSense model
def deepSense(inputs, train, reuse=False, name='deepSense'):
	with tf.variable_scope(name, reuse=reuse) as scope:
		used = tf.sign(tf.reduce_max(tf.abs(inputs), reduction_indices=2)) #(BATCH_SIZE, WIDE)
		length = tf.reduce_sum(used, reduction_indices=1) #(BATCH_SIZE)
		length = tf.cast(length, tf.int64)

		mask = tf.sign(tf.reduce_max(tf.abs(inputs), reduction_indices=2, keep_dims=True))
		mask = tf.tile(mask, [1,1,INTER_DIM]) # (BATCH_SIZE, WIDE, INTER_DIM)
		avgNum = tf.reduce_sum(mask, reduction_indices=1) #(BATCH_SIZE, INTER_DIM)

		# Reading sensor input, split into different modes

		# inputs shape =(BATCH_SIZE, WIDE, FEATURE_DIM)
		sensor_inputs = tf.expand_dims(inputs, axis=3)
		# sensor_inputs shape (BATCH_SIZE, WIDE, FEATURE_DIM, CHANNEL=1)
		radar_inputs, lidar_inputs = tf.split(sensor_inputs, num_or_size_splits=2, axis=2)

		# Individual layers, same architecture for each mode
		# Radar mode
		radar_conv1 = layers.convolution2d(radar_inputs, CONV_NUM, kernel_size=[1, 2*3*CONV_LEN],
						stride=[1, 2*3], padding='VALID', activation_fn=None, data_format='NHWC', scope='radar_conv1')
		radar_conv1 = batch_norm_layer(radar_conv1, train, scope='radar_BN1')
		radar_conv1 = tf.nn.relu(radar_conv1)
		radar_conv1_shape = radar_conv1.get_shape().as_list()
		radar_conv1 = layers.dropout(radar_conv1, CONV_KEEP_PROB, is_training=train,
			noise_shape=[radar_conv1_shape[0], 1, 1, radar_conv1_shape[3]], scope='radar_dropout1')

		radar_conv2 = layers.convolution2d(radar_conv1, CONV_NUM, kernel_size=[1, CONV_LEN_INTE],
						stride=[1, 1], padding='VALID', activation_fn=None, data_format='NHWC', scope='radar_conv2')
		radar_conv2 = batch_norm_layer(radar_conv2, train, scope='radar_BN2')
		radar_conv2 = tf.nn.relu(radar_conv2)
		radar_conv2_shape = radar_conv2.get_shape().as_list()
		radar_conv2 = layers.dropout(radar_conv2, CONV_KEEP_PROB, is_training=train,
			noise_shape=[radar_conv2_shape[0], 1, 1, radar_conv2_shape[3]], scope='radar_dropout2')

		radar_conv3 = layers.convolution2d(radar_conv2, CONV_NUM, kernel_size=[1, CONV_LEN_LAST],
						stride=[1, 1], padding='VALID', activation_fn=None, data_format='NHWC', scope='radar_conv3')
		radar_conv3 = batch_norm_layer(radar_conv3, train, scope='radar_BN3')
		radar_conv3 = tf.nn.relu(radar_conv3)
		radar_conv3_shape = radar_conv3.get_shape().as_list()
		radar_conv_out = tf.reshape(radar_conv3, [radar_conv3_shape[0], radar_conv3_shape[1], 1, radar_conv3_shape[2],radar_conv3_shape[3]])

		# Lidar mode
		lidar_conv1 = layers.convolution2d(lidar_inputs, CONV_NUM, kernel_size=[1, 2 * 3 * CONV_LEN],
										  stride=[1, 2 * 3], padding='VALID', activation_fn=None, data_format='NHWC',
										  scope='lidar_conv1')
		lidar_conv1 = batch_norm_layer(lidar_conv1, train, scope='lidar_BN1')
		lidar_conv1 = tf.nn.relu(lidar_conv1)
		lidar_conv1_shape = lidar_conv1.get_shape().as_list()
		lidar_conv1 = layers.dropout(lidar_conv1, CONV_KEEP_PROB, is_training=train,
									noise_shape=[lidar_conv1_shape[0], 1, 1, lidar_conv1_shape[3]], scope='lidar_dropout1')

		lidar_conv2 = layers.convolution2d(lidar_conv1, CONV_NUM, kernel_size=[1, CONV_LEN_INTE],
										  stride=[1, 1], padding='VALID', activation_fn=None, data_format='NHWC',
										  scope='lidar_conv2')
		lidar_conv2 = batch_norm_layer(lidar_conv2, train, scope='lidar_BN2')
		lidar_conv2 = tf.nn.relu(lidar_conv2)
		lidar_conv2_shape = lidar_conv2.get_shape().as_list()
		lidar_conv2 = layers.dropout(lidar_conv2, CONV_KEEP_PROB, is_training=train,
									noise_shape=[lidar_conv2_shape[0], 1, 1, lidar_conv2_shape[3]], scope='lidar_dropout2')

		lidar_conv3 = layers.convolution2d(lidar_conv2, CONV_NUM, activation_fn=None, kernel_size=[1, CONV_LEN_LAST],
										  stride=[1, 1], padding='VALID', data_format='NHWC', scope='lidar_conv3')
		lidar_conv3 = batch_norm_layer(lidar_conv3, train, scope='lidar_BN3')
		lidar_conv3 = tf.nn.relu(lidar_conv3)
		lidar_conv3_shape = lidar_conv3.get_shape().as_list()
		lidar_conv_out = tf.reshape(lidar_conv3, [lidar_conv3_shape[0], lidar_conv3_shape[1], 1, lidar_conv3_shape[2],
												lidar_conv3_shape[3]])

		# End of individual layers
		# Merge layers

		sensor_conv_in = tf.concat([radar_conv_out, lidar_conv_out], 2)
		senor_conv_shape = sensor_conv_in.get_shape().as_list()
		sensor_conv_in = layers.dropout(sensor_conv_in, CONV_KEEP_PROB, is_training=train,
			noise_shape=[senor_conv_shape[0], 1, 1, 1, senor_conv_shape[4]], scope='sensor_dropout_in')

		sensor_conv1 = layers.convolution2d(sensor_conv_in, CONV_NUM2, kernel_size=[1, 2, CONV_MERGE_LEN],
						stride=[1, 1, 1], padding='SAME', activation_fn=None, data_format='NDHWC', scope='sensor_conv1')
		sensor_conv1 = batch_norm_layer(sensor_conv1, train, scope='sensor_BN1')
		sensor_conv1 = tf.nn.relu(sensor_conv1)
		sensor_conv1_shape = sensor_conv1.get_shape().as_list()
		sensor_conv1 = layers.dropout(sensor_conv1, CONV_KEEP_PROB, is_training=train,
			noise_shape=[sensor_conv1_shape[0], 1, 1, 1, sensor_conv1_shape[4]], scope='sensor_dropout1')

		sensor_conv2 = layers.convolution2d(sensor_conv1, CONV_NUM2, kernel_size=[1, 2, CONV_MERGE_LEN2],
						stride=[1, 1, 1], padding='SAME', activation_fn=None, data_format='NDHWC', scope='sensor_conv2')
		sensor_conv2 = batch_norm_layer(sensor_conv2, train, scope='sensor_BN2')
		sensor_conv2 = tf.nn.relu(sensor_conv2)
		sensor_conv2_shape = sensor_conv2.get_shape().as_list()
		sensor_conv2 = layers.dropout(sensor_conv2, CONV_KEEP_PROB, is_training=train,
			noise_shape=[sensor_conv2_shape[0], 1, 1, 1, sensor_conv2_shape[4]], scope='sensor_dropout2')

		sensor_conv3 = layers.convolution2d(sensor_conv2, CONV_NUM2, kernel_size=[1, 2, CONV_MERGE_LEN3],
						stride=[1, 1, 1], padding='SAME', activation_fn=None, data_format='NDHWC', scope='sensor_conv3')
		sensor_conv3 = batch_norm_layer(sensor_conv3, train, scope='sensor_BN3')
		sensor_conv3 = tf.nn.relu(sensor_conv3)
		sensor_conv3_shape = sensor_conv3.get_shape().as_list()
		sensor_conv_out = tf.reshape(sensor_conv3, [sensor_conv3_shape[0], sensor_conv3_shape[1], sensor_conv3_shape[2]*sensor_conv3_shape[3]*sensor_conv3_shape[4]])

		# Recurrent layers

		gru_cell1 = tf.contrib.rnn.GRUCell(INTER_DIM)
		if train:
			gru_cell1 = tf.contrib.rnn.DropoutWrapper(gru_cell1, output_keep_prob=0.5)

		gru_cell2 = tf.contrib.rnn.GRUCell(INTER_DIM)
		if train:
			gru_cell2 = tf.contrib.rnn.DropoutWrapper(gru_cell2, output_keep_prob=0.5)

		cell = tf.contrib.rnn.MultiRNNCell([gru_cell1, gru_cell2])
		init_state = cell.zero_state(BATCH_SIZE, tf.float32)

		cell_output, final_stateTuple = tf.nn.dynamic_rnn(cell, sensor_conv_out, sequence_length=length, initial_state=init_state, time_major=False)

		sum_cell_out = tf.reduce_sum(cell_output*mask, axis=1, keep_dims=False)
		avg_cell_out = sum_cell_out/avgNum

		# Output layer

		logits = layers.fully_connected(avg_cell_out, OUT_DIM, activation_fn=None, scope='output')

		return logits

# Preprocessing
raw_files = os.listdir(DIR_RAW)
label_file = os.listdir(DIR_LABELS)[0]
dp.preprocess(raw_data_files=raw_files, raw_label_file=label_file, processed_file_name=PROCESSED_FILE, dir_raw=DIR_RAW, dir_label=DIR_LABELS, stride=BATCH_SIZE)

# Training phase setup
global_step = tf.Variable(0, trainable=False)

csvFileList = [DIR_PROCESSED + 'processed_data.csv']
batch_feature, batch_label = input_pipeline(csvFileList, BATCH_SIZE)
logits = deepSense(batch_feature, True, name='deepSense')
predict = logits[1] # Regression task, 1-dim tensor
batchLoss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_label)
loss = tf.reduce_mean(batchLoss)

# Evaluation phase setup
# csvEvalFileList = [DIR_PROCESSED + 'processed_eval.csv']
# batch_eval_feature, batch_eval_label = input_pipeline(csvEvalFileList, BATCH_SIZE, shuffle_sample=False)
# logits_eval = deepSense(batch_eval_feature, False, reuse=True, name='deepSense')
# predict_eval = tf.argmax(logits_eval, axis=1)
# loss_eval = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_eval, labels=batch_eval_label))

# Loss regularization
t_vars = tf.trainable_variables()
regularizers = 0.
for var in t_vars:
	regularizers += tf.nn.l2_loss(var)
loss += 5e-4 * regularizers

# Adam algorithm for back propagation
discOptimizer = tf.train.AdamOptimizer(
		learning_rate=1e-4,
		beta1=0.5,
		beta2=0.9
	).minimize(loss, var_list=t_vars)

# Iterating
with tf.Session() as sess:
	tf.global_variables_initializer().run()
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	for iteration in range(TOTAL_ITER_NUM):
		_, lossV, _trainY, _predict = sess.run([discOptimizer, loss, batch_label, predict])
		_label = _trainY[1]
		_accuracy = np.mean(_label == _predict)

		# Training phase visualization
		plt.plot('iterations', 'train cross entropy', data=lossV)
		plt.plot('iterations', 'train accuracy', data=_accuracy)

		# Supervised learning, evaluate (test) at every 50 training batches (optional)
		# if iteration % 50 == 49:
		# 	dev_anguracy = []
		# 	dev_cross_entropy = []
		# 	for eval_idx in range(EVAL_ITER_NUM):
		# 		# eval_loss_v, _trainY, _predict = sess.run([loss, trainY, predict], feed_dict ={train_status: False})
		# 		eval_loss_v, _trainY, _predict = sess.run([loss, batch_eval_label, predict_eval])
		# 		_label = np.argmax(_trainY, axis=1)
		# 		_anguracy = np.mean(_label == _predict)
		# 		dev_anguracy.append(_anguracy)
		# 		dev_cross_entropy.append(eval_loss_v)