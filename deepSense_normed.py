import tensorflow as tf

# CNN hidden layers configuration
CONV_LEN = 3
CONV_LEN_INTE = 4
CONV_LEN_LAST = 5
CONV_NUM = 64
CONV_MERGE_LEN = 8
CONV_MERGE_LEN2 = 6
CONV_MERGE_LEN3 = 4
CONV_NUM2 = 64
CONV_KEEP_PROB = 0.8
RNN_WINDOW_1 = 5
RNN_WINDOW_2 = 10
INTER_DIM = 120
RNN_KEEP_PROB = 0.5
OUT_DIM = 24

MODALITIES = 3
TWO_F = 40
IMG_SHAPE = 57600 #57600
LIDAR_SHAPE = 16 #70
RADAR_SHAPE = 16 #12
SHAPE = IMG_SHAPE + LIDAR_SHAPE + RADAR_SHAPE

# def individual_abs_err_0(y_true, y_pred): return tf.constant(abs(y_true[0] - y_pred[0]))
# def individual_abs_err_1(y_true, y_pred): return tf.constant(abs(y_true[1] - y_pred[1]))
# def individual_abs_err_2(y_true, y_pred): return tf.constant(abs(y_true[2] - y_pred[2]))


# DeepSense model
class deepSense(tf.keras.Model):

    def __init__(self):
        super(deepSense, self).__init__(name='deepSense')
        layers = tf.keras.layers
        # Individual layers, same architecture for each mode
        # First modality (image)
        self.conv1_img = layers.Conv2D(filters = CONV_NUM, kernel_size=[CONV_LEN, IMG_SHAPE],
                            strides=(1, 6), padding='valid', activation=None, input_shape=[-1, TWO_F, IMG_SHAPE, 1])
        self.bn1_img = layers.BatchNormalization()
        self.conv2_img = layers.Conv1D(filters=CONV_NUM, kernel_size=(CONV_LEN_INTE),
                                   strides=1, padding='valid', activation=None)
        self.bn2_img = layers.BatchNormalization()
        self.conv3_img = layers.Conv1D(filters=CONV_NUM, kernel_size=(CONV_LEN_LAST),
                                   strides=1, padding='valid', activation=None)
        self.bn3_img = layers.BatchNormalization()

        # Second modality (lidar)
        self.conv1_lidar = layers.Conv2D(filters=CONV_NUM, kernel_size=[CONV_LEN, LIDAR_SHAPE],
                                       strides=(1, 6), padding='valid', activation=None, input_shape=[-1, TWO_F, LIDAR_SHAPE, 1])
        self.bn1_lidar = layers.BatchNormalization()
        self.conv2_lidar = layers.Conv1D(filters=CONV_NUM, kernel_size=(CONV_LEN_INTE),
                                       strides=1, padding='valid', activation=None)
        self.bn2_lidar = layers.BatchNormalization()
        self.conv3_lidar = layers.Conv1D(filters=CONV_NUM, kernel_size=(CONV_LEN_LAST),
                                       strides=1, padding='valid', activation=None)
        self.bn3_lidar = layers.BatchNormalization()

        # Third modality (radar)
        self.conv1_radar = layers.Conv2D(filters=CONV_NUM, kernel_size=[CONV_LEN, RADAR_SHAPE],
                                       strides=(1, 6), padding='valid', activation=None, input_shape=[-1, TWO_F, RADAR_SHAPE, 1])
        self.bn1_radar = layers.BatchNormalization()
        self.conv2_radar = layers.Conv1D(filters=CONV_NUM, kernel_size=(CONV_LEN_INTE),
                                       strides=1, padding='valid', activation=None)
        self.bn2_radar = layers.BatchNormalization()
        self.conv3_radar = layers.Conv1D(filters=CONV_NUM, kernel_size=(CONV_LEN_LAST),
                                       strides=1, padding='valid', activation=None)
        self.bn3_radar = layers.BatchNormalization()

        # End of individual layers, merge layers start
        self.conv4 = layers.Conv2D(filters=CONV_NUM, kernel_size=[CONV_MERGE_LEN, MODALITIES],
                                   strides=(1, 1), padding='valid', activation=None)
        self.bn4 = layers.BatchNormalization()

        self.conv5 = layers.Conv1D(filters=CONV_NUM, kernel_size=(CONV_MERGE_LEN2),
                                   strides=1, padding='valid', activation=None)
        self.bn5 = layers.BatchNormalization()

        self.conv6 = layers.Conv1D(filters=CONV_NUM, kernel_size=(CONV_MERGE_LEN3),
                                   strides=1, padding='valid', activation=None)
        self.bn6 = layers.BatchNormalization()

        # Recurrent layers, (batch_size, time_steps, features)
        self.gru1 = layers.GRU(units=INTER_DIM)
        self.gru2 = layers.GRU(units=INTER_DIM)

        # Output layer, regression task
        self.out = layers.Dense(OUT_DIM)

    def call(self, x, training=True): # Test on: one time interval and one modality
        # print(x.shape)
        x0, x1, x2 = tf.split(x, [IMG_SHAPE, LIDAR_SHAPE, RADAR_SHAPE], axis=2)
        # Invidual layers for time intervals
        x0 = self.conv1_img(x0)
        x0 = self.bn1_img(x0)
        x0 = tf.nn.relu(x0)
        if training:
            x0 = tf.nn.dropout(x0, keep_prob=CONV_KEEP_PROB)
        shape = x0.get_shape().as_list()
        x0 = tf.reshape(x0, [-1, shape[1], shape[2] * shape[3]])
        # print("x0 conv1 finished")
        # print("x0 shape: ", x0.shape)

        x0 = self.conv2_img(x0)
        x0 = self.bn2_img(x0)
        x0 = tf.nn.relu(x0)
        if training:
            x0 = tf.nn.dropout(x0, keep_prob=CONV_KEEP_PROB)

        # print("x0 conv2 finished")
        # print("x0 shape: ", x0.shape)

        x0 = self.conv3_img(x0)
        x0 = self.bn3_img(x0)
        x0 = tf.nn.relu(x0)
        if training:
            x0 = tf.nn.dropout(x0, keep_prob=CONV_KEEP_PROB)
        # print("x0 conv3 finished")
        # print("x0 shape: ", x0.shape)

        x1 = self.conv1_lidar(x1)
        x1 = self.bn1_lidar(x1)
        x1 = tf.nn.relu(x1)
        if training:
            x1 = tf.nn.dropout(x1, keep_prob=CONV_KEEP_PROB)
        shape = x1.get_shape().as_list()
        x1 = tf.reshape(x1, [-1, shape[1], shape[2] * shape[3]])
        # print("x1 conv1 finished")
        # print("x1 shape: ", x1.shape)

        x1 = self.conv2_lidar(x1)
        x1 = self.bn2_lidar(x1)
        x1 = tf.nn.relu(x1)
        if training:
            x1 = tf.nn.dropout(x1, keep_prob=CONV_KEEP_PROB)
        # print("x1 conv2 finished")
        # print("x1 shape: ", x1.shape)

        x1 = self.conv3_lidar(x1)
        x1 = self.bn3_lidar(x1)
        x1 = tf.nn.relu(x1)
        if training:
            x1 = tf.nn.dropout(x1, keep_prob=CONV_KEEP_PROB)
        # print("x1 conv3 finished")
        # print("x1 shape: ", x1.shape)

        x2 = self.conv1_radar(x2)
        x2 = self.bn1_radar(x2)
        x2 = tf.nn.relu(x2)
        if training:
            x2 = tf.nn.dropout(x2, keep_prob=CONV_KEEP_PROB)
        shape = x2.get_shape().as_list()
        x2 = tf.reshape(x2, [-1, shape[1], shape[2] * shape[3]])
        # print("x2 conv1 finished")
        # print("x2 shape: ", x2.shape)

        x2 = self.conv2_radar(x2)
        x2 = self.bn2_radar(x2)
        x2 = tf.nn.relu(x2)
        if training:
            x2 = tf.nn.dropout(x2, keep_prob=CONV_KEEP_PROB)
        # print("x2 conv2 finished")
        # print("x2 shape: ", x2.shape)

        x2 = self.conv3_radar(x2)
        x2 = self.bn3_radar(x2)
        x2 = tf.nn.relu(x2)
        if training:
            x2 = tf.nn.dropout(x2, keep_prob=CONV_KEEP_PROB)
        # print("x2 conv3 finished")
        # print("x2 shape: ", x2.shape)

        # Flatten and concatenate modalities
        shape = x0.get_shape().as_list()
        x0 = tf.reshape(x0, [-1, shape[1] * shape[2]])
        shape = x1.get_shape().as_list()
        x1 = tf.reshape(x1, [-1, shape[1] * shape[2]])
        shape = x2.get_shape().as_list()
        x2 = tf.reshape(x2, [-1, shape[1] * shape[2]])
        x= tf.stack([x0, x1, x2])
        x = tf.transpose(x, perm=[1, 2, 0])
        shape = x.get_shape().as_list()
        x = tf.reshape(x, [-1, shape[1], shape[2], 1])

        # print("ready for merged, x shape:", x.shape)

        x = self.conv4(x)
        x = self.bn4(x)
        x = tf.nn.relu(x)
        if training:
            x = tf.nn.dropout(x, keep_prob=CONV_KEEP_PROB)
        shape = x.get_shape().as_list()
        x = tf.reshape(x, [-1, shape[1], shape[2] * shape[3]])
        # print("x conv4 finished")
        # print("x shape: ", x.shape)

        x = self.conv5(x)
        x = self.bn5(x)
        x = tf.nn.relu(x)
        if training:
            x = tf.nn.dropout(x, keep_prob=CONV_KEEP_PROB)
        # print("x conv5 finished")
        # print("x shape: ", x.shape)

        x = self.conv6(x)
        x = self.bn6(x)
        x = tf.nn.relu(x)
        if training:
            x = tf.nn.dropout(x, keep_prob=CONV_KEEP_PROB)
        # print("x conv6 finished")
        # print("x shape: ", x.shape)

        shape = x.get_shape().as_list()
        x = tf.reshape(x, [-1, shape[1] * shape[2]])
        shape = x.get_shape().as_list()
        x = tf.reshape(x, [-1, 1, shape[1]])
        # print("ready for rnn, x shape:", x.shape)

        x = self.gru1(x)
        if training:
            x = tf.nn.dropout(x, keep_prob=RNN_KEEP_PROB)
        shape = x.get_shape().as_list()
        x = tf.reshape(x, [-1, 1, shape[1]])
        # print("x gru1 finished")
        # print("x shape: ", x.shape)

        x = self.gru2(x)
        if training:
            x = tf.nn.dropout(x, keep_prob=RNN_KEEP_PROB)
        # print("x gru2 finished")
        # print("x shape: ", x.shape)

        x = self.out(x)
        return x

# Scipt testing purpose only
# from tensorflow.keras.callbacks import TensorBoard
# from time import time
# DIR = 'small_data/'
#
# normed_train_data = pd.read_csv(DIR + 'test_ds.csv',
#                       na_values = "?", comment='\t',
#                       sep=",", skipinitialspace=True, header=None).values
# normed_train_data = np.reshape(normed_train_data, (-1, TWO_F, SHAPE, 1))
#
# normed_train_label = pd.read_csv(DIR + 'test_ds_label.csv',
#                       na_values = "?", comment='\t',
#                       sep=",", skipinitialspace=True, header=None).values
#
# model = deepSense()
#
# model.compile(loss='mse', optimizer='adam',
#                   metrics=['mae', 'mse', 'accuracy',
#                            tf.keras.metrics.kullback_leibler_divergence,
#                            tf.keras.metrics.binary_crossentropy,
#                            ])
# tensorboard = TensorBoard(log_dir='logs/{}'.format(time()), histogram_freq=0, write_graph=True, write_images=True)
# saver = tf.train.Saver()
# checkpoint_file = DIR + "ds_checkpoint.ckpt"
#
# with tf.Session() as sess:
#     print("Entering session...")
#     history = model.fit(
#           x = normed_train_data, y = normed_train_label, batch_size=1, shuffle=False,
#           epochs=10,
#           validation_split=0, verbose=1, callbacks=[tensorboard]
#         )
#     hist = pd.DataFrame(history.history)
#     hist['epoch'] = history.epoch
#     hist.to_csv(DIR+"ds_test_epoch_hist.csv")
#     print("Saving checkpoint...")
#     save_path = saver.save(sess, checkpoint_file)
#     print("Checkpoint saved")