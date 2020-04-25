import numpy as np
import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from scipy import stats

from azureml.core import Run

print("Tensorflow version:", tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--batch-size', type=int, dest='batch_size', default=50, help='mini batch size for training')
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.001, help='learning rate')
parser.add_argument('--n-fc-layer', type=int, dest='n_fc_layer', default=16, help='number of neurons in fully connected layer')
parser.add_argument('--dropout-ratio', type=float, dest='dropout_ratio', default=0.1, help='ratio of dropout layer')

args = parser.parse_args()

data_folder = args.data_folder

print('training dataset is stored here:', data_folder)

datasets_filename_labels = [
    ('output_label_idle.txt', 'idle'),
    ('output_label_idle_2.txt', 'idle'),
    ('output_label_working.txt', 'working'),
    ('output_label_working_2.txt', 'working')
]

column_names = ['ax', 'ay', 'az']

def load_dataset(config):
    dataframes = []
    for filename, label in config:
        path = glob.glob(os.path.join(data_folder, '**/{}'.format(filename)), recursive=True)[0]
        dataframe = pd.read_csv(path, names=column_names)
        dataframe['label'] = label
        dataframes.append(dataframe)
    
    return pd.concat(dataframes)

def windows(data, size):
    start = 0
    while start < len(data):
        yield int(start), int(start + size)
        start += size

def segment_signals(data, window_size = 100):
    data_per_class = []
    
    for label in np.unique(data['label']):
        subset = data[data['label'] == label]
        length = len(subset)
        remaining = length % window_size
        max_length = length - remaining
        data_per_class.append(subset[:max_length])

    trimmed_data = pd.concat(data_per_class)
    
    result = []
    labels = []
    
    for (start, end) in windows(trimmed_data, window_size):
        row = []
        x_axes = data['ax'][start:end] / 1000.0
        y_axes = data['ay'][start:end] / 1000.0
        z_axes = data['az'][start:end] / 1000.0
        
        for x, y, z in zip(x_axes, y_axes, z_axes):
            row.append(x)
            row.append(y)
            row.append(z)
        row = np.array(row)
        row = row.reshape((100, 3))
        label = stats.mode(trimmed_data['label'][start:end])[0][0]
        
        result.append(row)
        labels.append(label)
    return np.array(result), np.array(labels)


dataset = load_dataset(datasets_filename_labels)
segments, labels = segment_signals(dataset, 100)
labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)

n_rows = segments.shape[1]
n_cols = segments.shape[2]
n_channels = 1
n_filters = 16
kernel_size = 2
pooling_window_size = 2
n_fc_layer= args.n_fc_layer
train_split_ratio = 0.8
epochs = 100
batch_size = args.batch_size
n_class = labels.shape[1]
dropout_ratio = args.dropout_ratio
learning_rate = args.learning_rate


# reshape data to fit the CNN model input shape
reshaped_segments = segments.reshape(segments.shape[0], n_rows, n_cols, 1)

# split training and test data
train_split = np.random.rand(len(reshaped_segments)) < train_split_ratio
train_x = reshaped_segments[train_split]
test_x = reshaped_segments[~train_split]
train_x = np.nan_to_num(train_x)
test_x = np.nan_to_num(test_x)
train_y = labels[train_split]
test_y = labels[~train_split]

print("Train X shape:", train_x.shape)
print("Train Y shape:", train_y.shape)

model = Sequential()
model.add(Conv2D(n_filters, (kernel_size, kernel_size), input_shape=(n_rows, n_cols, 1), activation='relu', padding='same'))
model.add(MaxPooling2D((pooling_window_size, pooling_window_size)))
model.add(Dropout(dropout_ratio))
model.add(Flatten())
model.add(Dense(n_fc_layer, activation='relu'))
model.add(Dropout(dropout_ratio))
model.add(Dense(n_class, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=Adagrad(learning_rate=learning_rate), metrics=['accuracy'])

# start an Azure ML run
run = Run.get_context()

class LogRunMetrics(Callback):
    # callback at the end of every epoch
    def on_epoch_end(self, epoch, log):
        # log a value repeated which creates a list
        run.log('Loss', log['loss'])
        run.log('Accuracy', log['accuracy'])
        run.log('Validation Loss', log['val_loss'])
        run.log('Validation Accuracy', log['val_accuracy'])

# start training
history = model.fit(train_x, train_y,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_split=1-train_split_ratio,
                    callbacks=[LogRunMetrics()])

# evaluate model
score = model.evaluate(test_x, test_y, verbose=0)

# log a single value
run.log("Final test loss", score[0])
print('Test loss:', score[0])

run.log('Final test accuracy', score[1])
print('Test accuracy:', score[1])

plt.figure(figsize=(6, 3))
plt.title('Motor Activity Recognition ({} epochs)'.format(epochs), fontsize=14)
plt.plot(history.history['accuracy'], 'b-', label='Accuracy', lw=4, alpha=0.5)
plt.plot(history.history['loss'], 'r--', label='Loss', lw=4, alpha=0.5)
plt.legend(fontsize=12)
plt.grid(True)

# log an image
run.log_image('Accuracy vs Loss', plot=plt)

# create a ./outputs/model folder in the compute target
# files saved in the "./outputs" folder are automatically uploaded into run history
os.makedirs('./outputs/model', exist_ok=True)

# serialize NN architecture to JSON
model_json = model.to_json()
# save model JSON
with open('./outputs/model/model.json', 'w') as f:
    f.write(model_json)
# save model weights
model.save_weights('./outputs/model/model.h5')
print("model saved in ./outputs/model folder")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model to disk
with open("./outputs/model/model.tflite", "wb") as f:
    f.write(tflite_model)