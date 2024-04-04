import numpy as np
import os
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from feature_extraction import extract_features, get_features

cnn_model_dir = '../models/emotion_recognition/emotion_cnn_model.h5'
lstm_model_dir = '../models/emotion_recognition/emotion_lstm_model.h5'
svm_model_dir = '../models/emotion_recognition/emotion_svm_model.joblib'
# Reuse or import your feature extraction and augmentation functions here
# For the sake of this example, functions are assumed to be imported

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

# Load Models
# svm_model = load(svm_model_dir)

# Encoder - assuming you have saved this during training or rebuild here
encoder = OneHotEncoder()
# Populate 'categories' with your emotions in the same order as during training
encoder.fit(np.array(['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']).reshape(-1, 1))

# Example usage
if __name__ == "__main__":
    test_audio_path = '../data/emotions_clips/audio_speech_actors_01-24/'
    if(os.path.isfile(cnn_model_dir)):
        CNN_Model = load_model(cnn_model_dir)
    if(os.path.isfile(lstm_model_dir)):
        LSTM_Model = load_model(lstm_model_dir)
    #if(os.path.isfile(svm_model_dir)):
    #    SVM_Model = load_model(svm_model_dir)

    cnn_prediction = predict_emotion_from_file(cnn_model, test_audio_path)
    lstm_prediction = predict_emotion_from_file(lstm_model, test_audio_path)
    
    print(f"Predicted Emotion (CNN): {cnn_prediction}")
    print(f"Predicted Emotion (LSTM): {lstm_prediction}")
    # Optionally, play the audio
    # ipd.Audio(test_audio_path)
