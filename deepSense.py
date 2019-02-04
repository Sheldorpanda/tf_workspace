import tensorflow as tf
import pandas as pd
import numpy as np
import os

# CNN hidden layers configuration
CONV_LEN = 3
CONV_LEN_INTE = 4
CONV_LEN_LAST = 5
CONV_NUM = 64
CONV_MERGE_LEN = 8
CONV_MERGE_LEN2 = 6
CONV_MERGE_LEN3 = 4
CONV_NUM2 = 64
INTER_DIM = 120
CONV_KEEP_PROB = 0.8
MODALITIES = 1

# Data sample configuration
FEATURE_DIM = 4 # Total dimension of multimodal inputs
BATCH_SIZE = 64 # Batch size of training data, also the strides for preprocessing
TOTAL_ITER_NUM = 10000 #1000000000
OUT_DIM = 3#len(idDict)

# DeepSense model
class deepSense(tf.keras.Model):

    def __init__(self):
        super(deepSense, self).__init__(name='deepSense')
        layers = tf.keras.layers

        # Individual layers, same architecture for each mode
        # First modality (image)
        self.conv1 = layers.Conv2D(filters = CONV_NUM, kernel_size=[CONV_LEN, 1],
                            strides=(1, 6), padding='valid', activation=None, input_shape=(40, 230400, 1))
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv1D(filters=CONV_NUM, kernel_size=(CONV_LEN_INTE),
                                   strides=1, padding='valid', activation=None)
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv1D(filters=CONV_NUM, kernel_size=(CONV_LEN_LAST),
                                   strides=1, padding='valid', activation=None)
        self.bn3 = layers.BatchNormalization()

        # Second modality (lidar)

        # Third modality (radar)

        # End of individual layers, merge layers start
        self.conv4 = layers.Conv2D(filters=CONV_NUM, kernel_size=[MODALITIES, CONV_MERGE_LEN],
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
        # self.gru2 = layers.GRU(units=INTER_DIM)

        # Output layer, regression task
        self.out = layers.Dense(OUT_DIM)

    def call(self, inputs, training=True): # Test on: one time interval and one modality
        print(inputs)
        print(inputs.shape)
        # Invidual layers for time intervals
        X = []
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        if training:
            x = tf.nn.dropout(x, keep_prob=CONV_KEEP_PROB)
        shape = x.get_shape().as_list()
        x = tf.reshape(x, [-1, shape[1], shape[2] * shape[3]])

        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)
        if training:
            x = tf.nn.dropout(x, keep_prob=CONV_KEEP_PROB)

        x = self.conv3(x)
        x = self.bn3(x)
        x = tf.nn.relu(x)
        if training:
            x = tf.nn.dropout(x, keep_prob=CONV_KEEP_PROB)
        X.append(x)

        X = np.array(X)

        print(X)

        # Flatten and concatenate modalities
        # x = tf.keras.layers.Flatten(X)
        x = tf.expand_dims(x, axis=-1) # for 1 modality

        x = self.conv4(x)
        x = self.bn4(x)
        x = tf.nn.relu(x)
        if training:
            x = tf.nn.dropout(x, keep_prob=CONV_KEEP_PROB)
        shape = x.get_shape().as_list()
        x = tf.reshape(x, [-1, shape[1], shape[2] * shape[3]])

        x = self.conv5(x)
        x = self.bn5(x)
        x = tf.nn.relu(x)
        if training:
            x = tf.nn.dropout(x, keep_prob=CONV_KEEP_PROB)

        x = self.conv6(x)
        x = self.bn6(x)
        x = tf.nn.relu(x)
        if training:
            x = tf.nn.dropout(x, keep_prob=CONV_KEEP_PROB)

        x = self.gru1(x)
        if training:
            x = tf.nn.dropout(x, keep_prob=0.5)
        # x = self.gru2(x)
        # if training:
        #     x = tf.nn.dropout(x, keep_prob=0.5)

        x = self.out(x)
        return x

    def loss(model, x, y):
        y_ = model(x)
        return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

    def build_model(train_data):
        model = deepSense()

        # discOptimizer = tf.train.AdamOptimizer(
        #     learning_rate=1e-4,
        #     beta1=0.5,
        #     beta2=0.9
        # ).minimize(deepSense.loss(model, train_data, train_label))

        optimizer = tf.keras.optimizers.RMSprop(0.001)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse', 'accuracy'])
        return model

dir = "sensor_data/LCA_Real_Life_Highway__511 sec/LCA_Real-Life_Highway_1_Ford_Fiesta_1/processed/"

file_data = dir + "processed.csv"
train_data = pd.read_csv(file_data,
                      na_values = "?", comment='\t',
                      sep=",", skipinitialspace=True)
file_label = dir + "indicators.csv"
label_cols = ["range", "azimuth", "dist_l"]
train_label = pd.read_csv(file_label, names=label_cols,
                      na_values = "?", comment='\t',
                      sep=",", skipinitialspace=True)
# print(train_data)
# print(train_label)

train_data = np.expand_dims(np.expand_dims(train_data, axis=0), axis=3) # samples, rows, cols, channels

model = deepSense().build_model()

with tf.Session() as sess:
    print("Entering session...")
    history = model.fit(
      x = train_data, y = train_label.values,
      epochs=1, validation_split = 0.2, verbose=0,
    )
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist)
    hist.to_csv(dir+"ds_test_hist.csv")
    # save_path = saver.save(sess, dir + "control_model_test.ckpt")

model.summary()