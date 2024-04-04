import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def CreateSVMModel(x_train, y_train, x_test, y_test):
    # # SVM Model
    # svm_model = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True)
    # # SVM Model Training
    # svm_model.fit(x_train.reshape(x_train.shape[0], -1), np.argmax(y_train, axis=1))  # Reshape x_train for SVM and use argmax to convert y_train back from one-hot encoding
    # # SVM Model Evaluation
    # y_pred_svm = svm_model.predict(x_test.reshape(x_test.shape[0], -1))
    # y_pred_proba_svm = svm_model.predict_proba(x_test.reshape(x_test.shape[0], -1))  # Probability estimates for ROC curve, etc.
    # # Since SVM predicts class labels, convert y_test from one-hot to labels for comparison
    # y_test_labels = np.argmax(y_test, axis=1)
    # # Predict labels with the SVM model
    # y_pred_svm = svm_model.predict(x_test.reshape(x_test.shape[0], -1))

    # Train the SVM model
    model = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True)
    model.fit(x_train.reshape(x_train.shape[0], -1), np.argmax(y_train, axis=1))  # Reshape x_train for SVM and use argmax to convert y_train back from one-hot encoding
    x_train_svm = x_train.reshape(x_train.shape[0], -1)  # Reshaping for SVM
    y_train_svm = np.argmax(y_train, axis=1)  # Converting from one-hot to labels
    param_grid = {
        'C': [0.1, 1, 10, 100], 
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=3)
    grid.fit(x_train_svm, y_train_svm)
    # print("Best Parameters: ", grid.best_params_)
    # Use the best model from grid search for predictions and further evaluations
    svm_model_best = grid.best_estimator_
    # Evaluate the SVM model on the test set
    y_pred_svm = model.predict(x_test.reshape(x_test.shape[0], -1))
    y_pred_proba_svm = model.predict_proba(x_test.reshape(x_test.shape[0], -1))  # Probability estimates for ROC curve, etc.
    # Since SVM predicts class labels, convert y_test from one-hot to labels for comparison
    y_test_labels = np.argmax(y_test, axis=1)
    # Predict labels with the SVM model
    y_pred_svm = model.predict(x_test.reshape(x_test.shape[0], -1))
    return model