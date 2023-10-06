import xgboost as xgb


class MLClassifier:
    def __init__(self, model='', parameters=''):
        self.parameters = parameters
        self.model = model

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        # self.__model = eval(f'{model}()')
        depth = self.parameters['max_depth']
        learning_rate = self.parameters['learning_rate']
        n_estimators = self.parameters['n_estimators']
        subsample = self.parameters['subsample']
        colsample_bytree = self.parameters['colsample_bytree']
        self.__model = xgb.XGBModel(n_estimators=n_estimators, max_depth=depth, learning_rate=learning_rate,
                                    subsample=subsample, colsample_bytree=colsample_bytree)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        prediction = self.model.predict(X)
        return prediction


class LeapClassifier:
    CARDIO_DISEASES = [
        'I00',
        'I01',
        'I05',
        'I06',
        'I07',
        'I08',
        'I09',
        'I10',
        'I11',
        'I12',
        'I13',
        'I15',
        'I16',
        'I20',
        'I21',
        'I22',
        'I23',
        'I24',
        'I25',
        'I26',
        'I27',
        'I28',
        'I30',
        'I31',
        'I33',
        'I34',
        'I35',
        'I36',
        'I37',
        'I38',
        'I39',
        'I40',
        'I42',
        'I43',
        'I44',
        'I45',
        'I46',
        'I47',
        'I48',
        'I49',
        'I50',
        'I51',
        'I52',
        'I60',
        'I61',
        'I62',
        'I63',
        'I65',
        'I66',
        'I67',
        'I68',
        'I69',
        'I70',
        'I71',
        'I72',
        'I73',
        'I74',
        'I75',
        'I77',
        'I78',
        'I79',
        'I80',
        'I81',
        'I82',
        'I83',
        'I85',
        'I86',
        'I87',
        'I88',
        'I89',
        'I95',
        'I96',
        'I97',
        'I99']

    def __init__(self, data, convert_prediction: bool):
        self.convert_prediction = convert_prediction
        self.data = data

    def predict(self):
        points = 0

        # Age points
        age = int(self.data.get('PresentAge', 0))
        if 40 <= age <= 49:
            points += 1
        elif 50 <= age <= 59:
            points += 2
        elif age > 60:
            points += 3

        # Gender points
        gender = self.data.get('Gender', '')
        if gender == 0:
            points += 1

        # Diabetes during pregnancy points
        if gender == 1 and self.data.get('O24', '') == 'true':
            points += 1

        # First degree relatives with Diabetes points
        relatives_diabetes = self.data.get('DiabeteshistoryofFamily', '')
        if relatives_diabetes == 1:
            points += 1

        # History of hypertension points
        hypertension = self.data.get('I10', '')
        if hypertension == 1:
            points += 1

        # Physical activity points
        # loa = self.data.get('LOA', '')
        # if loa == 0:
        #     points += 1

        # Race points
        race = self.data.get('RaceCategory', '')
        if race == 1:
            points += 1

        # Cardio disease history
        for i in self.CARDIO_DISEASES:
            cardio = self.data.get(i)
            if cardio == 1:
                points += 1
                break

        # Polycystic Ovaries history
        ovaries = self.data.get('E28')
        if ovaries == 1:
            points += 1

        # BMI Points
        bmi = self.data.get('BMI')
        if 25 <= bmi < 30:
            points += 1
        elif 30 <= bmi < 40:
            points += 2
        elif bmi >= 40:
            points += 3

        # Lab results points
        hdl = float(self.data.get('HdlResult', 0))
        tg = float(self.data.get('TgResult', 0))
        hba1c = float(self.data.get('HbResult', 0))
        fbs = float(self.data.get('FbsResult', 0))

        if hdl < 35:
            points += 1
        if tg > 250:
            points += 1
        if hba1c > 5.7:
            points += 1
        if 100 <= fbs <= 125:
            points += 1

        if self.convert_prediction:
            return True if points > 5 else False
        else:
            return points

    def recommend_remedy(self):
        pass
