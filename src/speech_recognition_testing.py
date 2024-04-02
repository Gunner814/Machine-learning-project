import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

# Step 2: Define a function to extract MFCC features from an audio file
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}, {e}")
        return None
    return mfccs.T  # Transpose to match LSTM input requirements

# Step 3: Define a function to predict the transcription
def predict_transcription(model, file_path):
    features = extract_features(file_path)
    if features is None:
        return ""
    # Adjust the shape of features as required by your model
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    prediction = model.predict(features)
    
    # Convert prediction to text (placeholder, implement based on your project)
    predicted_text = "..."  # You need to implement the logic based on how you encoded the labels
    return predicted_text

@tf.keras.utils.register_keras_serializable()
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

# Step 1: Load the trained model
model_path = '../models/speech_recognition_model.keras'
#model = load_model(model_path, safe_mode=False)
#model = tf.keras.models.load_model(model_path, custom_objects={'ctc_lambda_func': ctc_lambda_func}, safe_mode=False)
model = tf.keras.models.load_model(model_path, safe_mode=False)

# Step 4: Predict transcription for a new audio file and save to text file
audio_file_path = '../data/test_clips/test.mp3'  # Update with the path to your audio file
predicted_text = predict_transcription(model, audio_file_path)

# Save the predicted transcription to a text file
with open('../data/predicted_transcriptions.txt', 'w') as file:
    file.write(predicted_text)

print("Prediction saved to predicted_transcriptions.txt")
