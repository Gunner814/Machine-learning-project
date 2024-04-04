from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, LeakyReLU
from data_preprocessing import load_data

def CreateCNNModel(x_train):
    # CNN modelling
    # model=Sequential()
    # model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(x_train.shape[1], 1)))
    # model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
    # model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
    # model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
    # model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
    # model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
    # model.add(Dropout(0.2))
    # model.add(Conv1D(32, kernel_size=5, strides=1, padding='same', activation='relu'))
    # model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
    # model.add(Flatten())
    # model.add(Dense(units=16, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(units=7, activation='softmax'))
    # model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, padding='same', activation=LeakyReLU(alpha=0.1), input_shape=(x_train.shape[1], 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Conv1D(128, kernel_size=3, padding='same', activation=LeakyReLU(alpha=0.1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation=LeakyReLU(alpha=0.1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(units=7, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# # Load and preprocess data
# RAV = '../data/emotions_clips/audio_speech_actors_01-24/'
# df = load_data(RAV)
# X, Y = get_features(df)

# # Model definition
# model = Sequential([
#     Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(X.shape[1], 1)),
#     MaxPooling1D(pool_size=5, strides=2, padding='same'),
#     # Add more layers...
#     Flatten(),
#     Dense(7, activation='softmax')
# ])

# # Compile, train, and evaluate model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# # model.fit(...)
