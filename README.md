# Machine-learning-project
English Language Identification Using NLP

Project Team 20

Members:

Tan Jian Jie

Celine Leong Jia Yi

Anderson Phua Tai Dah

Chua Zhen Xun

# GitHub Link
https://github.com/Gunner814/Machine-learning-project

# User Manual
1. Run the MLbuilddependencies.bat file to install all dependencies.
2. Run the program in the command line "python emotion_recognition_training.py" and all 3 models will be trained and tested.
3. At the end of the program, Accuracy of prediction of CNN and LSTM models will be displayed in the data/results folder and console.  There will also be a graph showing the Loss and Accuracy of the LSTM and CNN Models during the training process as the number of epochs increase.

# Disclaimer
Our group has changed the project scope from our initial proposal. This project now focuses on emotion recognition from audio inputs.

# Background
Emotion recognition from audio signals is a challenging and fascinating area that sits at the crossroads of psychology, computer science, and artificial intelligence. It aims to enable computers to understand and interpret human emotions expressed through voice, enhancing human-computer interaction.

# Project Overview
This project aims to develop a system capable of identifying human emotions from audio inputs, leveraging deep learning models for accurate emotion recognition. Our approach focuses on training models that can accurately identify a range of emotions, enhancing applications in areas such as interactive voice response systems, mental health assessment, and virtual assistants.

# Data
Our dataset comprises audio clips expressing a wide range of emotions. These recordings serve as the primary training material for our models, focusing on achieving high precision in emotion recognition for improved human-computer interaction.

Data processing: 
1. Noise Injection
2. Time Stretching
3. Shifting
4. Pitch Shifting


# Algorithms
We explore three advanced algorithmic approaches for emotion recognition:

Convolutional Neural Network (CNN): Known for its robust feature extraction capabilities, particularly useful for identifying patterns in audio spectrograms that correspond to different emotions.

Long Short-Term Memory (LSTM): A type of Recurrent Neural Network (RNN) suited for sequential data, making it highly effective for tasks requiring an understanding of temporal relationships in audio data.

Support Vector Machine (SVM): A machine learning model that finds the optimal hyperplane for classifying data points into different categories, used here to classify extracted features into emotion categories.

# Justificaiton for Model Choices
CNNs are chosen for their unparalleled pattern recognition abilities, crucial for processing and identifying the complex features within audio data that signify different emotions.

LSTMs are utilized for their proficiency in handling sequential and time-series data, essential for capturing the temporal dynamics in speech that convey emotional states.

SVMs provide a reliable classification technique that complements deep learning models by offering a different approach to feature space partitioning, which is useful for fine-grained emotion classification.

# Timeline

Week 8: Proposal documentation.

Week 9: Collection and preprocessing of English audio data, initiation of CNN model development.

Week 10: Training of the CNN model, initial accuracy evaluation, and feedback identification.

Week 11: Model optimization for enhanced accuracy and efficiency, performance re-evaluation.

Week 12: Final model testing, project result review, and documentation.

Week 13: Preparation and finalization of video and presentation materials.