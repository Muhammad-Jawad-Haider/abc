from sklearn.model_selection import GridSearchCV
from classifier import MLClassifier
from classifier_evaluator import ClassifierEvaluator
from classifier import LeapClassifier
import numpy as np
import pickle


class Controller:
    def __init__(self, grid_params=''):
        self.grid_params = grid_params
        self.model = MLClassifier()
        self.grid = GridSearchCV(MLClassifier(), self.grid_params, cv=10, scoring='accuracy')

    def tune_model(self, X_train, y_train):
        self.grid.fit(X_train, y_train)
        self.model = MLClassifier(parameters=self.grid.best_estimator_)
        self.model.fit(X_train, y_train)
        print('Model trained with parameters: ', self.grid.best_estimator_)

    def save_model(self, filepath):
        pickle.dump(self.model, open(filepath, "wb"))
        print(f'Model successfully save at {filepath}.')

    def load_model(self, filepath):
        self.model = pickle.load(open(filepath, "rb"))
        print(f'Model successfully loaded from {filepath}.')

    def predict(self, X_test, y_test):
        self.y_pred = self.model.predict(X_test)
        return self.y_pred

    def predict_by_leap(self, data):
        y_pred = np.zeros(len(data))
        y_remedy = np.zeros(len(data))
        for i in range(len(data)):
            classifier = LeapClassifier(data.iloc[i].to_dict(), convert_prediction=True)
            y_pred[i] = classifier.predict()
            y_remedy[i] = classifier.recommend_remedy()

        return y_pred, y_remedy

    def evaluate_model(self, y_test, y_pred):
        evaluator = ClassifierEvaluator(y_test, y_pred)
        evaluation = {
            'f1_score': evaluator.calculate_f1_score(),
            'accuracy': evaluator.calculate_accuracy(),
            'confusion_matrix': evaluator.make_confusion_matrix()
        }

        return evaluation
