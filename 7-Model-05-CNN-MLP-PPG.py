# -*- coding: utf-8 -*-
"""
Automatically generated by Colab.
"""

from google.colab import drive
drive.mount('/content/drive')

# Required Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import os
from sklearn.metrics import precision_score, recall_score

# Enable GPU
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print(f'Found GPU at: {device_name}')

# section2: Load Data
signal_data = pd.read_csv("/content/drive/My Drive/Colab Notebooks/downsampled_preprocessed_flagged_PPG_diminished.csv")
emotions = pd.read_csv("/content/drive/My Drive/Colab Notebooks/emotion_dimension_normalized_Shifted.csv")

# section3: Data Preparation
valid_signal_data = signal_data[signal_data['Data_Origin'] == 1]
X_signal = valid_signal_data.loc[:, 'Data_1':'Data_200'].values
y = emotions.loc[valid_signal_data.index, ['Polarity', 'Complexity', 'Intensity']].values

X_signal = X_signal[..., np.newaxis]

X_signal_train, X_signal_val, y_train, y_val = train_test_split(
    X_signal, y, test_size=0.2, random_state=42)

# Create Model with MLP Ensemble for Classification
def create_model(input_shape_signal, num_ensembles=10, dropout_rate=0.5):
    cnn_input = layers.Input(shape=input_shape_signal)

    x = layers.Conv1D(512, kernel_size=50, strides=1, activation='relu')(cnn_input)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(256, kernel_size=25, strides=1, activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(128, kernel_size=20, activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)

    combined_input = x  # No concatenate needed since there's only one input

    ensemble_outputs = []
    for _ in range(num_ensembles):
        mlp = layers.Dense(729)(combined_input)
        mlp = layers.BatchNormalization()(mlp)
        mlp = layers.Activation('relu')(mlp)
        mlp = layers.Dropout(dropout_rate)(mlp)

        mlp = layers.Dense(486)(mlp)
        mlp = layers.BatchNormalization()(mlp)
        mlp = layers.Activation('relu')(mlp)

        mlp = layers.Dense(324)(mlp)
        mlp = layers.Activation('relu')(mlp)

        mlp = layers.Dense(216)(mlp)
        mlp = layers.Dropout(dropout_rate)(mlp)

        mlp = layers.Dense(144)(mlp)
        mlp = layers.BatchNormalization()(mlp)
        mlp = layers.Activation('relu')(mlp)

        mlp = layers.Dense(96)(mlp)
        mlp = layers.Dropout(dropout_rate)(mlp)

        ensemble_outputs.append(layers.Dense(3, activation='relu')(mlp))

    combined_output = layers.average(ensemble_outputs)

    model = models.Model(inputs=cnn_input, outputs=combined_output)
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse')
    return model

input_shape_signal = (200, 1)
model = create_model(input_shape_signal, num_ensembles=10, dropout_rate=0.5)

# Function to Update CSV with Metrics
csv_path = '/content/drive/My Drive/Colab Notebooks/Models/learning_curves.csv'

def update_csv(epoch, train_loss, val_loss, train_acc, val_acc,
               polarity_train_loss, complexity_train_loss, intensity_train_loss,
               polarity_val_loss, complexity_val_loss, intensity_val_loss,
               polarity_train_acc, complexity_train_acc, intensity_train_acc,
               polarity_val_acc, complexity_val_acc, intensity_val_acc,
               train_precision, val_precision, train_recall, val_recall,
               polarity_train_precision, complexity_train_precision, intensity_train_precision,
               polarity_val_precision, complexity_val_precision, intensity_val_precision,
               polarity_train_recall, complexity_train_recall, intensity_train_recall,
               polarity_val_recall, complexity_val_recall, intensity_val_recall):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        last_epoch = df['Epoch'].max()
        new_epoch = last_epoch + 1
    else:
        new_epoch = epoch

    new_row = pd.DataFrame({
        'Epoch': [epoch],
        'Train Loss': [train_loss],
        'Val Loss': [val_loss],
        'Train Accuracy': [train_acc],
        'Val Accuracy': [val_acc],
        'Polarity Train Loss': [polarity_train_loss],
        'Complexity Train Loss': [complexity_train_loss],
        'Intensity Train Loss': [intensity_train_loss],
        'Polarity Val Loss': [polarity_val_loss],
        'Complexity Val Loss': [complexity_val_loss],
        'Intensity Val Loss': [intensity_val_loss],
        'Polarity Train Accuracy': [polarity_train_acc],
        'Complexity Train Accuracy': [complexity_train_acc],
        'Intensity Train Accuracy': [intensity_train_acc],
        'Polarity Val Accuracy': [polarity_val_acc],
        'Complexity Val Accuracy': [complexity_val_acc],
        'Intensity Val Accuracy': [intensity_val_acc],
        'Train Precision': [train_precision],
        'Val Precision': [val_precision],
        'Train Recall': [train_recall],
        'Val Recall': [val_recall],
        'Polarity Train Precision': [polarity_train_precision],
        'Complexity Train Precision': [complexity_train_precision],
        'Intensity Train Precision': [intensity_train_precision],
        'Polarity Val Precision': [polarity_val_precision],
        'Complexity Val Precision': [complexity_val_precision],
        'Intensity Val Precision': [intensity_val_precision],
        'Polarity Train Recall': [polarity_train_recall],
        'Complexity Train Recall': [complexity_train_recall],
        'Intensity Train Recall': [intensity_train_recall],
        'Polarity Val Recall': [polarity_val_recall],
        'Complexity Val Recall': [complexity_val_recall],
        'Intensity Val Recall': [intensity_val_recall]
    })

    if not os.path.exists(csv_path):
        new_row.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(csv_path, index=False)

# Training Plot Callback
class TrainingPlot(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val):
        super(TrainingPlot, self).__init__()
        self.train_loss = []
        self.val_loss = []
        self.X_val = X_val
        self.y_val = y_val
        self.polarity_acc = []
        self.complexity_acc = []
        self.intensity_acc = []
        self.train_acc = []
        self.val_acc = []
        self.train_precision = []
        self.val_precision = []
        self.train_recall = []
        self.val_recall = []
        self.polarity_train_precision = []
        self.complexity_train_precision = []
        self.intensity_train_precision = []
        self.polarity_val_precision = []
        self.complexity_val_precision = []
        self.intensity_val_precision = []
        self.polarity_train_recall = []
        self.complexity_train_recall = []
        self.intensity_train_recall = []
        self.polarity_val_recall = []
        self.complexity_val_recall = []
        self.intensity_val_recall = []

    def on_epoch_end(self, epoch, logs=None):
        self.train_loss.append(logs['loss'])
        self.val_loss.append(logs['val_loss'])

        y_train_pred = self.model.predict(self.model.train_data)
        y_val_pred = self.model.predict(self.X_val)

        if np.any(np.isnan(y_train_pred)) or np.any(np.isnan(y_val_pred)):
            print("NaN values found in predictions. Stopping training.")
            self.model.stop_training = True
            return

        y_train_pred_binary = np.round(y_train_pred).astype(int)
        y_val_pred_binary = np.round(y_val_pred).astype(int)

        polarity_train_acc = accuracy_score(np.round(self.model.train_labels[:, 0]), np.round(y_train_pred[:, 0]))
        complexity_train_acc = accuracy_score(np.round(self.model.train_labels[:, 1]), np.round(y_train_pred[:, 1]))
        intensity_train_acc = accuracy_score(np.round(self.model.train_labels[:, 2]), np.round(y_train_pred[:, 2]))
        train_acc = (polarity_train_acc + complexity_train_acc + intensity_train_acc) / 3

        polarity_val_acc = accuracy_score(np.round(self.y_val[:, 0]), np.round(y_val_pred[:, 0]))
        complexity_val_acc = accuracy_score(np.round(self.y_val[:, 1]), np.round(y_val_pred[:, 1]))
        intensity_val_acc = accuracy_score(np.round(self.y_val[:, 2]), np.round(y_val_pred[:, 2]))
        val_acc = (polarity_val_acc + complexity_val_acc + intensity_val_acc) / 3

        polarity_train_loss = mean_squared_error(self.model.train_labels[:, 0], y_train_pred[:, 0])
        complexity_train_loss = mean_squared_error(self.model.train_labels[:, 1], y_train_pred[:, 1])
        intensity_train_loss = mean_squared_error(self.model.train_labels[:, 2], y_train_pred[:, 2])

        polarity_val_loss = mean_squared_error(self.y_val[:, 0], y_val_pred[:, 0])
        complexity_val_loss = mean_squared_error(self.y_val[:, 1], y_val_pred[:, 1])
        intensity_val_loss = mean_squared_error(self.y_val[:, 2], y_val_pred[:, 2])

        polarity_train_precision = precision_score(np.round(self.model.train_labels[:, 0]), y_train_pred_binary[:, 0], zero_division=1, average='macro')
        complexity_train_precision = precision_score(np.round(self.model.train_labels[:, 1]), y_train_pred_binary[:, 1], zero_division=1, average='macro')
        intensity_train_precision = precision_score(np.round(self.model.train_labels[:, 2]), y_train_pred_binary[:, 2], zero_division=1, average='macro')
        train_precision = (polarity_train_precision + complexity_train_precision + intensity_train_precision) / 3

        polarity_val_precision = precision_score(np.round(self.y_val[:, 0]), y_val_pred_binary[:, 0], zero_division=1, average='macro')
        complexity_val_precision = precision_score(np.round(self.y_val[:, 1]), y_val_pred_binary[:, 1], zero_division=1, average='macro')
        intensity_val_precision = precision_score(np.round(self.y_val[:, 2]), y_val_pred_binary[:, 2], zero_division=1, average='macro')
        val_precision = (polarity_val_precision + complexity_val_precision + intensity_val_precision) / 3

        polarity_train_recall = recall_score(np.round(self.model.train_labels[:, 0]), y_train_pred_binary[:, 0], zero_division=1, average='macro')
        complexity_train_recall = recall_score(np.round(self.model.train_labels[:, 1]), y_train_pred_binary[:, 1], zero_division=1, average='macro')
        intensity_train_recall = recall_score(np.round(self.model.train_labels[:, 2]), y_train_pred_binary[:, 2], zero_division=1, average='macro')
        train_recall = (polarity_train_recall + complexity_train_recall + intensity_train_recall) / 3

        polarity_val_recall = recall_score(np.round(self.y_val[:, 0]), y_val_pred_binary[:, 0], zero_division=1, average='macro')
        complexity_val_recall = recall_score(np.round(self.y_val[:, 1]), y_val_pred_binary[:, 1], zero_division=1, average='macro')
        intensity_val_recall = recall_score(np.round(self.y_val[:, 2]), y_val_pred_binary[:, 2], zero_division=1, average='macro')
        val_recall = (polarity_val_recall + complexity_val_recall + intensity_val_recall) / 3

        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)
        self.train_precision.append(train_precision)
        self.val_precision.append(val_precision)
        self.train_recall.append(train_recall)
        self.val_recall.append(val_recall)

        self.polarity_train_precision.append(polarity_train_precision)
        self.complexity_train_precision.append(complexity_train_precision)
        self.intensity_train_precision.append(intensity_train_precision)
        self.polarity_val_precision.append(polarity_val_precision)
        self.complexity_val_precision.append(complexity_val_precision)
        self.intensity_val_precision.append(intensity_val_precision)
        self.polarity_train_recall.append(polarity_train_recall)
        self.complexity_train_recall.append(complexity_train_recall)
        self.intensity_train_recall.append(intensity_train_recall)
        self.polarity_val_recall.append(polarity_val_recall)
        self.complexity_val_recall.append(complexity_val_recall)
        self.intensity_val_recall.append(intensity_val_recall)

        print(f"Epoch {epoch + 1} - Polarity Loss (Val): {polarity_val_loss:.3f}, Complexity Loss (Val): {complexity_val_loss:.3f}, Intensity Loss (Val): {intensity_val_loss:.3f}")
        print(f"Epoch {epoch + 1} - Polarity ACC (Val): {polarity_val_acc:.3f}, Complexity ACC (Val): {complexity_val_acc:.3f}, Intensity ACC (Val): {intensity_val_acc:.3f}")
        print(f"Epoch {epoch + 1} - Total Loss (Val): {logs['val_loss']:.3f}, Total ACC (Val): {val_acc:.3f}")
        print(f"Epoch {epoch + 1} - Train Precision: {train_precision:.3f}, Val Precision: {val_precision:.3f}, Train Recall: {train_recall:.3f}, Val Recall: {val_recall:.3f}")

        update_csv(epoch + 1, logs['loss'], logs['val_loss'], train_acc, val_acc,
                   polarity_train_loss, complexity_train_loss, intensity_train_loss,
                   polarity_val_loss, complexity_val_loss, intensity_val_loss,
                   polarity_train_acc, complexity_train_acc, intensity_train_acc,
                   polarity_val_acc, complexity_val_acc, intensity_val_acc,
                   train_precision, val_precision, train_recall, val_recall,
                   polarity_train_precision, complexity_train_precision, intensity_train_precision,
                   polarity_val_precision, complexity_val_precision, intensity_val_precision,
                   polarity_train_recall, complexity_train_recall, intensity_train_recall,
                   polarity_val_recall, complexity_val_recall, intensity_val_recall)

        epochs = range(1, epoch + 2)
        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.train_loss, label='Training Loss')
        plt.plot(epochs, self.val_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.val_precision, label='Validation Precision')
        plt.plot(epochs, self.val_recall, label='Validation Recall')
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')
        plt.legend()
        plt.title('Validation Precision and Recall')

        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.train_acc, label='Training Accuracy')
        plt.plot(epochs, self.val_acc, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')

        plt.tight_layout()
        plt.show()

# Custom ModelCheckpoint Callback
class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        self.epochs_between_saves = 20
        self.epochs_since_last_save = 0
        super(CustomModelCheckpoint, self).__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.epochs_between_saves:
            self.epochs_since_last_save = 0
            super(CustomModelCheckpoint, self).on_epoch_end(epoch, logs)

checkpoint_callback = CustomModelCheckpoint(
    filepath='/content/drive/My Drive/Colab Notebooks/Models/Model-05-CNN-MLP-PPG_{epoch}.keras',
    save_best_only=False,
    save_weights_only=False,
    mode='auto',
    save_freq='epoch'
)

# Training the Model
training_plot = TrainingPlot(X_signal_val, y_val)
model.train_data = X_signal_train
model.train_labels = y_train

model.fit(
    X_signal_train, y_train,
    epochs=3000,
    batch_size=64,
    validation_data=(X_signal_val, y_val),
    callbacks=[training_plot, checkpoint_callback]
)

model.save('/content/drive/My Drive/Colab Notebooks/Models/Model-05-CNN-MLP-PPG.keras')
plt.ioff()