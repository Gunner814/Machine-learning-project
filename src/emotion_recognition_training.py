import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import warnings

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from data_preprocessing import load_data
from feature_extraction import get_features
from model_cnn import CNN_Model
from model_lstm import LSTM_Model
from model_svm import SVM_Model
from utils import ShowLossAndAccuracy

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

RAV = '../data/emotions_clips/audio_speech_actors_01-24/'
RAV_df = load_data(RAV)
RAV_df.labels.value_counts()

print(RAV_df.head())
print(RAV_df.describe())

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
X, Y, feature = [], [], []
for path, emotion in zip(RAV_df.path, RAV_df.emotion):
    feature = get_features(path)
    for ele in feature:
        X.append(ele)
        # appending emotion 3 times as we have made 3 augmentation techniques on each audio file.
        Y.append(emotion)

len(X), len(Y), RAV_df.path.shape
Features = pd.DataFrame(X)
Features['labels'] = Y
Features.to_csv('../data/results/features.csv', index=False)

print(Features.head())
print(Features.describe())

X = Features.iloc[: ,:-1].values
Y = Features['labels'].values

encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)

    # CNN_model_dir = '../models/emotion_recognition/emotion_cnn_model.h5'
    # # Create CNN Model
    # CNN = CNN_Model(x_train)
    # CNN.Summary()
    # # CNN Model Training
    # CNN.Train(x_train, y_train, x_test, y_test, batch_size=64, epochs=50)
    # # CNN Model Evaluation
    # CNN.Evaluate(x_test, y_test, encoder)
    # # Save CNN Model
    # CNN.Save(CNN_model_dir)

    # LSTM_model_dir = '../models/emotion_recognition/emotion_lstm_model.h5'
    # # Create LSTM Model
    # LSTM = LSTM_Model(x_train)
    # LSTM.Summary()
    # # LSTM Model Training
    # LSTM.Train(x_train, y_train, x_test, y_test, batch_size=64, epochs=50)
    # # LSTM Model Evaluation
    # LSTM.Evaluate(x_test, y_test, encoder, verbose=0)
    # # Save LSTM Model
    # #LSTM.Save(LSTM_model_dir) # Not working

SVM_model_dir = '../models/emotion_recognition/emotion_svm_model.joblib'
# Create SVM model
SVM = SVM_Model(x_train, y_train, x_test, y_test)
# SVM Model Training
SVM.Train()
# SVM Model Evaluation
SVM.Evaluate(encoder)
# Save SVM Model
SVM.Save(SVM_model_dir)

plt.rcParams.update({'font.size': 12})
epochs = [i for i in range(50)]
ShowLossAndAccuracy(epochs, CNN)
ShowLossAndAccuracy(epochs, LSTM)