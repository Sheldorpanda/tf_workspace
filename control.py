import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

def build_model_3(train_data):
    layers = tf.keras.layers
    model = tf.keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_data.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(2)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse', 'accuracy'])
    return model

def build_model_6(train_data):
    layers = tf.keras.layers
    model = tf.keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_data.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(2)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse', 'accuracy'])
    return model

# def plot_history(hist):
#     # hist = pd.DataFrame(history.history)
#     # hist['epoch'] = history.epoch
#
#     plt.figure()
#     plt.xlabel('Epoch')
#     plt.ylabel('Mean Abs Error')
#     plt.plot(hist['epoch'], hist['mean_absolute_error'],
#              label='Train Error')
#     plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
#              label='Val Error')
#     plt.legend()
#     plt.ylim([0, 5])
#
#     plt.figure()
#     plt.xlabel('Epoch')
#     plt.ylabel('Mean Square Error')
#     plt.plot(hist['epoch'], hist['mean_squared_error'],
#              label='Train Error')
#     plt.plot(hist['epoch'], hist['val_mean_squared_error'],
#              label='Val Error')
#     plt.legend()
#     plt.ylim([0, 20])

dir = "sensor_data/LCA_Real_Life_Highway__511 sec/LCA_Real-Life_Highway_1_Ford_Fiesta_1/"
file_data = dir + "i_s_ground_truth.csv"
file_label = dir + "a_ground_truth.csv"
data_cols = ["range", "azimuth", "dist_left", "velocity"]
label_cols = ["acceleration", "steering"]

train_data = pd.read_csv(file_data, names=data_cols,
                      na_values = "?", comment='\t',
                      sep=",", skipinitialspace=True)
train_label = pd.read_csv(file_label, names=label_cols,
                      na_values = "?", comment='\t',
                      sep=",", skipinitialspace=True)

data = train_data.copy()
label = train_label.copy()

train_stats = data.describe()
train_stats = train_stats.transpose()

normed_train_data = norm(data)

# sns.pairplot(data[data_cols], diag_kind="kde")
# plt.show()
print(normed_train_data)
print(label)

model = build_model_6(normed_train_data)
# model.summary()

saver = tf.train.Saver()

with tf.Session() as sess:
    history = model.fit(
      x = normed_train_data, y = label,
      epochs=2000, validation_split = 0.2, verbose=0,
    )
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist)
    hist.to_csv(dir+"control_model_6_hist.csv")
    save_path = saver.save(sess, dir + "control_model_6.ckpt")