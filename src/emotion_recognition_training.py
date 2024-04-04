import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import IPython.display as ipd
import plotly.express as px
import sys
import warnings
import seaborn as sns

from keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import dump, load

if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from feature_extraction import get_features, extract_features
from model_cnn import CreateCNNModel
from model_lstm import CreateLSTMModel
from model_svm import CreateSVMModel
from utils import ShowLossAndAccuracy

def create_waveplot(data, sr, e):
    plt.figure(figsize=(10, 3))
    plt.title('Waveplot for audio with {} emotion'.format(e), size=15)
    librosa.display.waveshow(data, sr=sr, color="red")
    plt.show()

def create_spectrogram(data, sr, e):
    # stft function converts the data into short term fourier transform
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 3))
    plt.title('Spectrogram for audio with {} emotion'.format(e), size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')   
    #librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate = rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

sns.set_style('darkgrid')
RAV = '../data/emotions_clips/audio_speech_actors_01-24/'
dir_list = os.listdir(RAV)

emotion, gender, path, feature = [], [] ,[] ,[]
for i in dir_list:
    fname = os.listdir(RAV + i)
    for f in fname:
        part = f.split('.')[0].split('-')
        emotion.append(int(part[2]))
        temp = int(part[6])
        if temp%2 == 0:
            temp = "female"
        else:
            temp = "male"
        gender.append(temp)
        path.append(RAV + i + '/' + f)

RAV_df = pd.DataFrame(emotion)
RAV_df = RAV_df.replace({1:'neutral', 2:'neutral', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'})
RAV_df = pd.concat([pd.DataFrame(gender),RAV_df],axis=1)
RAV_df.columns = ['gender','emotion']
RAV_df['labels'] =RAV_df.gender + '_' + RAV_df.emotion
RAV_df['source'] = 'RAVDESS'
RAV_df = pd.concat([RAV_df,pd.DataFrame(path, columns = ['path'])],axis=1)
RAV_df = RAV_df.drop(['gender'], axis=1)
RAV_df.labels.value_counts()

print(RAV_df.head())
print(RAV_df.describe())

px_fig = px.histogram(RAV_df, x='emotion', color='emotion', marginal='box',  
                      title='Emotion Count')
px_fig.update_layout(bargap=0.2)
px_fig.show()

px_fig = px.histogram(RAV_df, x='labels', color='emotion', marginal='box',  
                      title='Label Count')
px_fig.update_layout(bargap=0.2)
px_fig.show()

emotion='fear'

path = np.array(RAV_df.path[RAV_df.emotion==emotion])[1]
data, sampling_rate = librosa.load(path)
#create_waveplot(data, sampling_rate, emotion)
create_spectrogram(data, sampling_rate, emotion)
ipd.Audio(path)

# taking any example and checking for techniques.
path = np.array(RAV_df.path)[1]
data, sample_rate = librosa.load(path)
plt.figure(figsize=(14,4))
#librosa.display.waveshow(y=data, sr=sample_rate, color="blue")
ipd.Audio(path)

# noise injection
x = noise(data)
plt.figure(figsize=(14,4))
#librosa.display.waveshow(y=x, sr=sample_rate, color="green")
ipd.Audio(x, rate=sample_rate)

# stretching
x = stretch(data)
plt.figure(figsize=(14,4))
#librosa.display.waveshow(y=x, sr=sample_rate, color="yellow")
ipd.Audio(x, rate=sample_rate)

# shifting
x = shift(data)
plt.figure(figsize=(14,4))
#librosa.display.waveshow(y=x, sr=sample_rate, color="purple")
ipd.Audio(x, rate=sample_rate)

# pitch
x = pitch(data,sample_rate)
plt.figure(figsize=(14,4))
#librosa.display.waveshow(y=x, sr=sample_rate, color="black")
ipd.Audio(x, rate=sample_rate)

# Data preparation
X, Y = [], []
for path, emotion in zip(RAV_df.path, RAV_df.emotion):
    feature = get_features(path)
    for ele in feature:
        X.append(ele)
        # appending emotion 3 times as we have made 3 augmentation techniques on each audio file.
        Y.append(emotion)

len(X), len(Y), RAV_df.path.shape
Features = pd.DataFrame(X)
Features['labels'] = Y
Features.to_csv('features.csv', index=False)

print(Features.head())
print(Features.describe())

X = Features.iloc[: ,:-1].values
Y = Features['labels'].values

encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

# CMM Model
CNN_Model = CreateCNNModel(x_train)
CNN_Model.summary()
# CNN Model Training
rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=4, min_lr=0.0000001)
history=CNN_Model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test), callbacks=[rlrp])
# CNN Model Evaluation
print("CNN Model Accuracy: " , CNN_Model.evaluate(x_test,y_test)[1]*100 , "%")
# Save CNN Model
CNN_Model.save('../models/emotion_recognition/emotion_cnn_model.h5')  # Save the CNN model

# LSTM Model
LSTM_Model = CreateLSTMModel(x_train)
LSTM_Model.summary()
# LSTM Model Training
history_lstm = LSTM_Model.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_test, y_test))
# LSTM Model Evaluation
score_lstm = LSTM_Model.evaluate(x_test, y_test, verbose=0)
print("LSTM Model Accuracy: {:.2f}%".format(score_lstm[1] * 100))
# Save LSTM Model
#LSTM_Model.save('../models/emotion_recognition/emotion_lstm_model.h5')

# Train the SVM model
SVM_Model = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True)
SVM_Model.fit(x_train.reshape(x_train.shape[0], -1), np.argmax(y_train, axis=1))  # Reshape x_train for SVM and use argmax to convert y_train back from one-hot encoding
x_train_svm = x_train.reshape(x_train.shape[0], -1)  # Reshaping for SVM
y_train_svm = np.argmax(y_train, axis=1)  # Converting from one-hot to labels
param_grid = {
    'C': [0.1, 1, 10, 100], 
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'poly', 'sigmoid']
}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=3)
grid.fit(x_train_svm, y_train_svm)
# print("Best Parameters: ", grid.best_params_)
# Use the best model from grid search for predictions and further evaluations
#svm_model_best = grid.best_estimator_
# Evaluate the SVM model on the test set
#y_pred_proba_svm = SVM_Model.predict_proba(x_test.reshape(x_test.shape[0], -1))  # Probability estimates for ROC curve, etc.
# Since SVM predicts class labels, convert y_test from one-hot to labels for comparison
y_test_labels = np.argmax(y_test, axis=1)
y_pred_svm = SVM_Model.predict(x_test.reshape(x_test.shape[0], -1))
# Calculate the accuracy
svm_accuracy = accuracy_score(y_test_labels, y_pred_svm)
# Print the accuracy
print("SVM Model Accuracy: {:.2f}%".format(svm_accuracy * 100))
# Classification report
print("SVM Classification Report:")
print(classification_report(y_test_labels, y_pred_svm))
# Save SVM Model
dump(SVM_Model, '../models/emotion_recognition/emotion_svm_model.joblib')

plt.rcParams.update({'font.size': 12})
epochs = [i for i in range(50)]
train_acc = history.history['accuracy']
train_loss = history.history['loss']
test_acc = history.history['val_accuracy']
test_loss = history.history['val_loss']
ShowLossAndAccuracy(epochs, history)

pred_test = CNN_Model.predict(x_test)
y_pred = encoder.inverse_transform(pred_test)
y_test = encoder.inverse_transform(y_test)

df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
df['Predicted Labels'] = y_pred.flatten()
df['Actual Labels'] = y_test.flatten()

print(df.head())
print(classification_report(y_test, y_pred))