import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from joblib import dump, load

class SVM_Model:
    def __init__(self, x_train, y_train, x_test, y_test):
        # Define parameter grid for SVM GridSearchCV
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        self.grid = GridSearchCV(SVC(probability=True), param_grid, refit=True, verbose=2, cv=3)
        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(x_train.reshape(x_train.shape[0], -1))
        self.y_train = y_train
        self.x_test = scaler.transform(x_test.reshape(x_test.shape[0], -1))
        self.y_test = y_test

    def Train(self):
        x_train_svm = self.x_train.reshape(self.x_train.shape[0], -1)
        y_train_svm = np.argmax(self.y_train, axis=1)
        self.grid.fit(x_train_svm, y_train_svm)
        self.model = self.grid.best_estimator_

    def Evaluate(self, encoder):
        x_test_svm = self.x_test.reshape(self.x_test.shape[0], -1)
        y_pred_svm = self.model.predict(x_test_svm)
        # Calculate accuracy
        self.score = accuracy_score(np.argmax(self.y_test, axis=1), y_pred_svm)
        print("SVM Model's Accuracy: " , self.score * 100, "%")
        # Classification report
        print("SVM Classification Report:")
        print(classification_report(np.argmax(self.y_test, axis=1), y_pred_svm))
        # y_true = encoder.inverse_transform(self.y_test)
        # y_pred = encoder.inverse_transform(y_pred_svm.reshape(-1, 1))

        # with open('../data/results/result_svm_model.txt', 'w') as f:
        #     f.write(f"LSTM Model Accuracy: {self.score:.2f}%\n")
        #     for i in range(len(y_true)):
        #         f.write(f"Expected: {y_true[i][0]}, Predicted: {y_pred[i][0]}\n")
        #     f.close()

    def Save(self, save_dir):
        dump(self.model, save_dir)

    def Load(self, save_dir):
        self.model = load(save_dir)