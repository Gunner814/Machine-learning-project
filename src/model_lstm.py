from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from data_preprocessing import load_data

def CreateLSTMModel(x_train):
    # model = Sequential()
    # model.add(LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    # model.add(Dropout(0.5))
    # model.add(LSTM(32))
    # model.add(Dropout(0.5))
    # model.add(Dense(7, activation='softmax'))  # Assuming 7 emotions as output classes
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=7, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# # Load and preprocess data
# RAV = '../data/emotions_clips/audio_speech_actors_01-24/'
# df = load_data(RAV)
# X, Y = get_features(df)

# # Model definition
# model_lstm = Sequential([
#     LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
#     Dropout(0.5),
#     LSTM(32),
#     Dropout(0.5),
#     Dense(7, activation='softmax')
# ])

# # Compile, train, and evaluate model
# model_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# # model_lstm.fit(...)
