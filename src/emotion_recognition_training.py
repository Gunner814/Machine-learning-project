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

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from feature_extraction import get_features, extract_features
from model_cnn import CNN_Model
from model_lstm import LSTM_Model
from model_svm import SVM_Model
from utils import ShowLossAndAccuracy

def create_waveplot(data, sr, e, x=None, color="red"):
    plt.figure(figsize=(14,4))
    plt.title('Waveplot for audio with {} emotion'.format(e), size=15)
    librosa.display.waveshow(data, sr=sr, color=color)
    if(x is not None):
        ipd.Audio(x, sr)

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
create_waveplot(data, sampling_rate, emotion, color="red")
create_spectrogram(data, sampling_rate, emotion)

# taking any example and checking for techniques.
path = np.array(RAV_df.path)[1]
data, sample_rate = librosa.load(path)

# noise injection
x = noise(data)
# stretching
x = stretch(data)
# shifting
x = shift(data)
# pitch
x = pitch(data,sample_rate)

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

CNN_model_dir = '../models/emotion_recognition/emotion_cnn_model.h5'
# Create CNN Model
CNN = CNN_Model(x_train)
CNN.Summary()
# CNN Model Training
CNN.Train(x_train, y_train, x_test, y_test, batch_size=64, epochs=50)
# CNN Model Evaluation
CNN.Evaluate(x_test, y_test)
# Save CNN Model
CNN.Save(CNN_model_dir)

LSTM_model_dir = '../models/emotion_recognition/emotion_lstm_model.h5'
# Create LSTM Model
LSTM = LSTM_Model(x_train)
LSTM.Summary()
# LSTM Model Training
LSTM.Train(x_train, y_train, x_test, y_test, batch_size=64, epochs=50)
# LSTM Model Evaluation
LSTM.Evaluate(x_test, y_test, encoder, verbose=0)
# Save LSTM Model
#LSTM.Save(LSTM_model_dir) # Not working

SVM_model_dir = '../models/emotion_recognition/emotion_svm_model.joblib'
# Create SVM model
SVM = SVM_Model(x_train, y_train, x_test, y_test)
# SVM Model Training
SVM.Train()
# SVM Model Evaluation
SVM.Evaluate()
# Save SVM Model
SVM.Save(SVM_model_dir)

plt.rcParams.update({'font.size': 12})
epochs = [i for i in range(50)]
ShowLossAndAccuracy(epochs, CNN)
ShowLossAndAccuracy(epochs, LSTM)

# pred_test = CNN.predict(x_test)
# y_pred = encoder.inverse_transform(pred_test)
# y_test = encoder.inverse_transform(y_test)

# df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
# df['Predicted Labels'] = y_pred.flatten()
# df['Actual Labels'] = y_test.flatten()

# print(df.head())
# print(classification_report(y_test, y_pred))