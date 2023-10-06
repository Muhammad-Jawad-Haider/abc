from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score,confusion_matrix,precision_recall_curve,auc
import pandas as pd
import pathlib
import pickle
# from load_dataset import DatasetLoader
import sqlite3
from configparser import ConfigParser
# from connector import Connector

config = ConfigParser()
config.read('config.ini')
fold_path = eval(config['Paths']['fold_path'])
# fold_path = '/Result/3month/'
features = eval(config['Classifiers']['features'])
# features = 
# db = Connector().get_database(eval(config['MongoDB']['database']), eval(config['MongoDB']['username']), eval(config['MongoDB']['password']))
# collection = db.eval(config['MongoDB']['collection'])

class ModelTrainer:
    
    def __init__(self, model, parameters):
        self._model = model
        self._parameters = parameters
        
        self._train_df = pd.DataFrame(columns=["Folds","Accuracy"])
        self._valid_df = pd.DataFrame(columns=["Folds","Accuracy","Precision","Recall","F1 Score","AUC","Specificity","Sensitivity","PPV","NPV","AUPR"])
        self._test_df = pd.DataFrame(columns=["Folds","Accuracy","Precision","Recall","F1 Score","AUC","Specificity","Sensitivity","PPV","NPV","AUPR"])
        self._report_df = pd.DataFrame(columns=["Folds","Accuracy","Precision","Recall","F1 Score","AUC","Specificity","Sensitivity","PPV","NPV","AUPR"])
        
        self.dict = {}
    
    # @property
    # def model(self):
    #     return self._model
    
    # @property
    # def train_df(self):
    #     return self._train_df
    
    # @property
    # def valid_df(self):
    #     return self._valid_df
    
    # @property
    # def test_df(self):
    #     return self._test_df
    
    # @property
    # def report_df(self):
    #     return self._report_df
    

    # @dict.setter
    # def dict(self, new_value):
    #     # Additional logic or validation can be performed here

    def train_model(self, cv, scoring_param, refit_param) -> object:
        
        #-------------------------Perform grid search for each model-----------------------
        clf = GridSearchCV(estimator = self._model, param_grid = self._parameters, cv = cv, 
                                scoring = scoring_param , refit = refit_param,return_train_score=True) # cv: number of splits that is needed for cross validation. By default None
        clf.fit(self.X_train, self.y_train)
        
        self.best_estimator = clf.best_estimator_

        return self.best_estimator


    def load_trainset(self, fold_number: int, scaler: str, scale_data: bool = False) -> tuple:
        '''fold number: ith number of fold
        scale_data: True or False
        scaler: MinMaxScaler or StandardScaler
        '''
        # train_data  = pd.read_csv(fold_path + "TrainingDataFold" + str(fold_number) + "v5.csv")
        self.X_train = pd.read_csv(fold_path + 'train_set_x.csv')
        self.y_train  = pd.read_csv(fold_path + 'train_set_y.csv')
        # print(train_data)
        if scale_data:
            if scaler == 'MinMaxScaler':
                self.scaler = MinMaxScaler()
                self.X_train = self.scaler.fit_transform(self.X_train)
                return self.X_train, self.y_train
            
            elif scaler == 'StandardScaler':    
                self.scaler = StandardScaler()
                self.X_train = self.scaler.fit_transform(self.X_train)
                return self.X_train, self.y_train
            
            return self.X_train, self.y_train
        
        return self.X_train, self.y_train


    def load_validset(self, fold_number, scale_data=False) -> tuple:
        # valid_data = pd.read_csv(fold_path + "ValidDataFold" + str(fold_number) + "v5.csv")

        # self.X_valid = valid_data.drop(valid_data.columns[-1],axis=1)
        # self.y_valid  = valid_data[valid_data.columns[-1]]

        if scale_data:
            self.X_valid = self.scaler.transform(self.X_valid)

            return self.X_valid, self.y_valid
        
        return self.X_valid, self.y_valid


    def load_testset(self, fold_number, scale_data=False) -> tuple:
        # test_data  = pd.read_csv(fold_path + "TestDataFold" + str(fold_number) + "v5.csv")

        self.X_test = pd.read_csv(fold_path + 'test_set_x.csv')
        self.y_test = pd.read_csv(fold_path + 'test_set_x.csv')
    
        if scale_data:
            self.X_test = self.scaler.transform(self.X_test)
            
            return  self.X_test, self.y_test
        
        return  self.X_test, self.y_test
    

    def save_results(self, fold_number):
        y_pred = self.best_estimator.predict(self.X_train)
        self._train_df.loc[len(self._train_df)] = ['Fold '+str(fold_number), self.best_estimator.score(self.X_train,self.y_train)]

        # y_pred = self.best_estimator.predict(self.X_valid)
        # specificity, sensitivity, ppv, npv, aupr = ModelTrainer.calculate_metrics(self.y_valid, y_pred)
        # self._valid_df.loc[len(self._valid_df)] = ['Fold '+str(fold_number), accuracy_score(self.y_valid,y_pred), precision_score(self.y_valid,y_pred),
        #                                             recall_score(self.y_valid,y_pred), f1_score(self.y_valid,y_pred), roc_auc_score(self.y_valid,y_pred),
        #                                             specificity, sensitivity, ppv, npv, aupr]
        
        y_pred = self.best_estimator.predict(self.X_test)
        specificity, sensitivity, ppv, npv, aupr = ModelTrainer.calculate_metrics(self.y_test, y_pred)
        self._test_df.loc[len(self._test_df)] = ['Fold '+str(fold_number), accuracy_score(self.y_test,y_pred), precision_score(self.y_test,y_pred),
                                                    recall_score(self.y_test,y_pred), f1_score(self.y_test,y_pred), roc_auc_score(self.y_test,y_pred),
                                                    specificity, sensitivity, ppv, npv, aupr]
    
        # self.dict[self.best_estimator] = [accuracy_score(self.y_test,y_pred),self.scaler]
        self.dict[self.best_estimator] = [accuracy_score(self.y_test,y_pred)]
        
        print(accuracy_score(self.y_test,y_pred))
        
        # data =  [
        #         {
        #             "algorithm": "Logistic Regression",
        #             "fold": 1,
        #             "train_accuracy": 0.85,
        #             "test_accuracy": 0.78,
        #             "validation_accuracy": 0.80,
        #             "validation_precision": 0.75,
        #             "validation_recall": 0.82,
        #         }
        #         ]
        # collection.insert(data)
    @staticmethod
    def calculate_metrics(y_true, y_pred_probs):
        # Convert probabilities to binary predictions using a threshold of 0.5
        y_pred = (y_pred_probs >= 0.5).astype(int)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp)

        # Sensitivity (True Positive Rate)
        sensitivity = tp / (tp + fn)

        # Positive Predictive Value (Precision)
        ppv = tp / (tp + fp)

        # Negative Predictive Value
        npv = tn / (tn + fn)

        # Precision-Recall curve to calculate AUPR
        precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
        aupr = auc(recall, precision)

        return specificity, sensitivity, ppv, npv, aupr

        

    def combine_results(self):
        cols1 = [('','Folds'),('Train','Accuracy')]
        cols2 = [('','Folds'),('Test','Accuracy'),('','Precision'),('','Recall'),('','F1 Score'),('','AUC'),('',"Specificity"),('',"Sensitivity"),('',"PPV"),('',"NPV"),('',"AUPR")]
        # cols3 = [('','Folds'),('Valid','Accuracy'),(' ','Precision'),(' ','Recall'),(' ','F1 Score'),(' ','AUC'),(' ',"Specificity"),(' ',"Sensitivity"),(' ',"PPV"),(' ',"NPV"),(' ',"AUPR")]

        self._train_df.columns = pd.MultiIndex.from_tuples(cols1)
        self._test_df.columns = pd.MultiIndex.from_tuples(cols2)
        # self._valid_df.columns = pd.MultiIndex.from_tuples(cols3)
        
        self._report_df = pd.merge(self._train_df, self._test_df)
        # self._report_df = pd.merge(self._train_df, self._valid_df)
        # self._report_df = pd.merge(self._report_df, self._test_df)

        mean_values = self._report_df.iloc[:, 1:].mean()
        
        # Append the mean values as a new row, excluding the first column
        self._report_df.loc['Mean'] = [None] + mean_values.tolist()
        self._report_df = self._report_df.round(3)
        
        return self._report_df

        
    def __del__(self):
        pass
        
    # def store_indb(self,Tp,Fn,Fp,Tn,rmse,mae,mse): #Stores Performance parameters in dB
    #     conn = sqlite3.connect("Database.db")
    #     cur = conn.cursor()
        
    #     cur.execute("""CREATE TABLE IF NOT EXISTS Results (id INTEGER PRIMARY KEY, Algorithm, Train_Accuracy float,
    #                  TestAccuracy float,Tn float, RMSE float, MAE float, MSE float)""")
    #     conn.commit()
    #     cur.execute("INSERT INTO Evaluation VALUES(NULL,?,?,?,?,?,?,?)",(Tp,Fn,Fp,Tn,rmse,mae,mse))
    #     conn.commit()
