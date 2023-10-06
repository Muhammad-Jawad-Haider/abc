from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
    confusion_matrix, precision_recall_curve, auc


class ClassifierEvaluator:
    def __init__(self, y_real, y_predict):
        self.y_real = y_real
        self.y_predict = y_predict

    def calculate_f1_score(self):
        return f1_score(self.y_real, self.y_predict)

    def calculate_accuracy(self):
        return accuracy_score(self.y_real, self.y_predict)

    def calculate_precision(self):
        return precision_score(self.y_real, self.y_predict)

    def calculate_recall(self):
        return recall_score(self.y_real, self.y_predict)

    def make_confusion_matrix(self):
        return confusion_matrix(self.y_real, self.y_predict)

    def calculate_precision_recall_curve(self):
        return precision_recall_curve(self.y_real, self.y_predict)

    def calculate_auc(self):
        return auc(self.y_real, self.y_predict)
