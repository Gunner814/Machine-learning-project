import numpy as np
import librosa
import librosa.display
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import IPython.display as ipd

# Reuse or import your feature extraction and augmentation functions here
# For the sake of this example, functions are assumed to be imported
# from feature_extraction import extract_features, get_features

# Load Models
cnn_model = load_model('../models/emotion_recognition/emotion_cnn_model.h5')
lstm_model = load_model('../models/emotion_recognition/emotion_lstm_model.h5')

# Encoder - assuming you have saved this during training or rebuild here
encoder = OneHotEncoder()
# Populate 'categories' with your emotions in the same order as during training
encoder.fit(np.array(['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']).reshape(-1, 1))

def predict_emotion_from_file(model, file_path):
    # Extract features using your previously defined function
    features = get_features(file_path)
    
    # Standardize features as done during training
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features.reshape(1, -1))  # Reshape if necessary
    
    # Predict
    predictions = model.predict(features_scaled)
    
    # Decode predicted label
    predicted_label = encoder.inverse_transform(predictions)
    return predicted_label

# Example usage
if __name__ == "__main__":
    test_audio_path = '../data/emotions_clips/audio_speech_actors_01-24/' 
    cnn_prediction = predict_emotion_from_file(cnn_model, test_audio_path)
    lstm_prediction = predict_emotion_from_file(lstm_model, test_audio_path)
    
    print(f"Predicted Emotion (CNN): {cnn_prediction}")
    print(f"Predicted Emotion (LSTM): {lstm_prediction}")
    # Optionally, play the audio
    # ipd.Audio(test_audio_path)
