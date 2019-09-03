import pandas as pd
import numpy as np
import seaborn as sns
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class IncomeClassifier:
    def __init__(self, dataset):
        
        sns.set(style='white', context='notebook', palette='deep')


        # Handle for Null Data
        dataset = dataset.fillna(np.nan)

        # Reformat Column We Are Predicting: 0 means less than 50K. 1 means greater than 50K.
        dataset['income']=dataset['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})

        # Fill Missing Category Entries
        dataset["workclass"] = dataset["workclass"].fillna("X")
        dataset["occupation"] = dataset["occupation"].fillna("X")
        dataset["native.country"] = dataset["native.country"].fillna("United-States")

        # Convert Sex value to 0 and 1
        dataset["sex"] = dataset["sex"].map({"Male": 0, "Female":1})

        # Create Married Column - Binary Yes(1) or No(0)
        dataset["marital.status"] = dataset["marital.status"].replace(['Never-married','Divorced','Separated','Widowed'], 'Single')
        dataset["marital.status"] = dataset["marital.status"].replace(['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'], 'Married')
        dataset["marital.status"] = dataset["marital.status"].map({"Married":1, "Single":0})
        dataset["marital.status"] = dataset["marital.status"].astype(int)

        # Drop the data you don't want to use
        dataset.drop(labels=["workclass","fnlwgt", "education","occupation","relationship","race","native.country"], axis = 1, inplace = True)

        ###################################################
        ##################### MODELING #####################
        ####################################################
        # Split-out Validation Dataset and Create Test Variables
        array = dataset.values
        X = array[:,0:7]
        Y = array[:,7]
        validation_size = 0.20
        seed = 7
        X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,
            test_size=validation_size,random_state=seed)

        ####################################################
        ################# FINALIZE MODEL ###################
        ####################################################

        random_forest = RandomForestClassifier(n_estimators=250,max_features=5)
        random_forest.fit(X_train, Y_train)
        self.random_forest = random_forest
        """predictions = random_forest.predict(X_validation)
        print("Accuracy: %s%%" % (100*accuracy_score(Y_validation, predictions)))

        testdata = [[86, 11, 1, 1, 77320, 16486, 86], [33, 11, 0, 0, 15067, 8702, 64], [29, 7, 1, 1, 34194, 52682, 83], [91, 8, 0, 1, 33880, 42557, 32], [24, 2, 0, 1, 37467, 5372, 38], [89, 2, 0, 1, 11827, 95082, 40], [51, 9, 1, 1, 35669, 59004, 34], [25, 7, 0, 0, 97356, 96552, 57], [47, 5, 0, 1, 15845, 46542, 64], [23, 11, 1, 0, 33168, 14552, 79]]
        predictions = random_forest.predict_proba(testdata)
        print predictions"""
    
    def predict_probabilities(self, record):
        return self.random_forest.predict_proba([record])[0]

