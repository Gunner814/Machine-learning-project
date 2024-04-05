import pandas as pd

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Bidirectional

from sklearn.metrics import classification_report

class LSTM_Model:
    def __init__(self, x_train):
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(x_train.shape[1], 1)))
        self.model.add(Dropout(0.5))
        self.model.add(Bidirectional(LSTM(64)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(units=7, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def Summary(self):
        self.model.summary()

    def Train(self, x_train, y_train, x_test, y_test, batch_size=64, epochs=50):
        self.history = self.model.fit(x_train, y_train, batch_size, epochs, validation_data=(x_test, y_test))
        
    def Evaluate(self, x_test, y_test, encoder, verbose=0):
        self.score = self.model.evaluate(x_test, y_test, verbose=0)[1] * 100
        pred_test = self.model.predict(x_test)
        y_pred = encoder.inverse_transform(pred_test)
        y_test = encoder.inverse_transform(y_test)
        df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
        df['Predicted Labels'] = y_pred.flatten()
        df['Actual Labels'] = y_test.flatten()
        print(df.head())
        print(classification_report(y_test, y_pred))
        print("LSTM Model's Accuracy: " , self.score, "%")

    def Save(self, save_dir):
        self.model.save(save_dir)

    def Load(self, save_dir):
        self.model = load_model(save_dir)