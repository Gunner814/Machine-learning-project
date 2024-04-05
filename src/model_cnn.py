from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, LeakyReLU
from data_preprocessing import load_data
import pickle

class CNN_Model:
    def __init__(self, x_train):
        self.model = Sequential()
        self.model.add(Conv1D(64, kernel_size=3, padding='same', activation=LeakyReLU(alpha=0.1), input_shape=(x_train.shape[1], 1)))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Dropout(0.25))
        self.model.add(Conv1D(128, kernel_size=3, padding='same', activation=LeakyReLU(alpha=0.1)))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation=LeakyReLU(alpha=0.1)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(units=7, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def Summary(self):
        self.model.summary()

    def Train(self, x_train, y_train, x_test, y_test, batch_size=64, epochs=50):
        rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=4, min_lr=0.0000001)
        self.history=self.model.fit(x_train, y_train, batch_size, epochs, validation_data=(x_test, y_test), callbacks=[rlrp])
        
    def Evaluate(self, x_test, y_test, encoder):
        self.score = self.model.evaluate(x_test, y_test)[1] * 100
        pred_test = self.model.predict(x_test)
        print("CNN Model's Accuracy: " , self.score, "%")
        # Inverse transform to get original labels
        y_true = encoder.inverse_transform(y_test)
        y_pred = encoder.inverse_transform(pred_test)
        with open('../data/results/result_cnn_model.txt', 'w') as f:
            f.write(f"CNN Model Accuracy: {self.score:.2f}%\n")
            for i in range(len(y_true)):
                f.write(f"Expected: {y_true[i][0]}, Predicted: {y_pred[i][0]}\n")
            f.close()

    def Save(self, save_dir):
        self.model.save(save_dir)

    def Load(self, save_dir):
        self.model = pickle.load(open(save_dir, 'rb'))
        self.score = self.model.score
        #self.history = self.model.history