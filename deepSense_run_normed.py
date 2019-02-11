import data_preprocess_normed
import deepSense_normed
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from time import time
from tensorflow.keras.callbacks import TensorBoard

TWO_F = 40
IMG_SHAPE = 57600
LIDAR_SHAPE = 16 #70
RADAR_SHAPE = 16 #12
SHAPE = IMG_SHAPE + LIDAR_SHAPE + RADAR_SHAPE
EPOCHS = 1000

# DIR : [code, sceanrio_dir0: [img_dir: [imgs], non_img_csv, label_csv, log], scenario_dir1:[], ..., checkpoint]

# Preprocessing
DIR =  "training_data/"
IMG_DIR = "training_data_imgs/"
non_img_file = "input.csv"
label_file = "labels.csv"
checkpoint_file = DIR + "ds_training_checkpoint.ckpt"

labels = pd.read_csv(DIR + label_file,
                      na_values = "?", comment='\t',
                      sep=",", skipinitialspace=True, header=None).values
print("Normalizing labels...")
normed_train_label = data_preprocess_normed.norm_2d(labels)
normed_train_label = normed_train_label

non_img_cols = ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8",
                "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8",
                "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16",
                "t9", "t10", "t11", "t12", "t13", "t14", "t15", "t16",
                ]
non_img_read = pd.read_csv(DIR + non_img_file, names=non_img_cols,
                      na_values = "?", comment='\t',
                      sep=",", skipinitialspace=True, header=None)
print("Normalizing angles ...")
angle_cols = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8",
              "t9", "t10", "t11", "t12", "t13", "t14", "t15", "t16"]

for col in angle_cols:
    n = len(non_img_read[col])
    for i in range(n):
        if non_img_read[col][i] > 180:
            non_img_read[col][i] -= 360

print("Angles normalized")
non_img_read = non_img_read.values

img_read = data_preprocess_normed.convert_all_images(DIR + IMG_DIR)
print("img_read shape: ", img_read.shape)
print("non_img_read shape: ", non_img_read.shape)
time_series = np.concatenate((img_read, non_img_read), axis=1)
print("shape after concat: ", time_series.shape)
time_series = data_preprocess_normed.norm_2d(time_series)
print("shape after normalization: ", time_series.shape)
f = data_preprocess_normed.transform(time_series)
print(f.shape)

print("Data already normalized in frequency domain, reshape to fit model...")
normed_train_data = np.reshape(f, (-1, TWO_F, SHAPE, 1))

print("Building model...")
model = deepSense_normed.deepSense()
model.compile(loss='mse', optimizer='adam',
                  metrics=['mae', 'mse', 'accuracy',
                           tf.keras.metrics.kullback_leibler_divergence,
                           tf.keras.metrics.binary_crossentropy,
                           ])
saver = tf.train.Saver()
tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))

with tf.Session() as sess:
    print("Entering session...")
    # dirs = [str(d[0]) + '/' for d in os.walk(DIR)]
    # for dir in dirs:
    if os.path.isfile(DIR + checkpoint_file):
        saver.restore(sess, DIR + checkpoint_file)
        print("Checkpoint restored")
    else:
        print("Checkpoint not found, randomly init DNN")
    history = model.fit(
          x = normed_train_data, y = normed_train_label, batch_size=1, shuffle=False,
          epochs=EPOCHS,
          validation_split=0, verbose=1, callbacks=[tensorboard]
        )
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.to_csv(DIR+"ds_test_epoch_hist.csv")
    print("Saving checkpoint...")
    save_path = saver.save(sess, DIR + checkpoint_file)
    print("Checkpoint saved")