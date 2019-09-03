import random as random
from classifiers.income.income_classifier import IncomeClassifier
from classifiers.digits.digits import NeuralNetwork
from synthesizer import Synthesizer
import pandas as pd
import numpy as np
import pickle
import csv

############################
###### INCOME DATASET ######
############################

def incomeRandomizeFunction():
    x_age = random.randint(0, 100)
    x_education = random.randint(0, 16)
    x_marital_status = random.randint(0, 1)
    x_sex = random.randint(0, 1)
    x_gain = random.randint(0, 100000)
    x_loss = random.randint(0, 100000)
    x_hpw = random.randint(0, 100)
    return [x_age, x_education, x_marital_status, x_sex, x_gain, x_loss, x_hpw]


print "Loading Income Classifier Data\n"
dataset = pd.read_csv("classifiers/income/adult.csv")
print "Intializing Classifier\n"
income_classifier = IncomeClassifier(dataset)
print "Intializing Training Data Synthesis\n"

synthesizer = Synthesizer(
    2, 7, 1, 100, 0.9, 5, incomeRandomizeFunction, income_classifier.predict_probabilities)
    #c,kmax,kmin,iter_max,conf_min,rej_max,randomizeFunction,predictProb
synthesizer.synthesize(10) #number of records

"""print "Testing probabilities..."
print "Class 0 means less than 50K. 1 means greater than 50K"
for c in range(2):
    dataset = list(csv.reader(open("training_class_"+str(c)+".csv")))
    print "\n\nProb dist for class" + str(c)
    for record in dataset:
        print income_classifier.predict_probabilities(record)"""

###########################
###### DIGITS DATASET #####
###########################

def digitsRandomizeFunction():
    x_temp = np.random.randint(0, 256, 28*28)
    #x_temp = np.random.normal(256/2, 256/2, 28*28)
    fac = 0.99 / 255
    return np.asfarray(x_temp) * fac + 0.01


"""   
with open("pickled_mnist.pkl", "rb") as fh:
    data = pickle.load(fh)
train_imgs = data[0]
test_imgs = data[1]
train_labels = data[2]
test_labels = data[3]
train_labels_one_hot = data[4]
test_labels_one_hot = data[5]
image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size

ANN = NeuralNetwork(no_of_in_nodes = image_pixels, 
                    no_of_out_nodes = 10, 
                    no_of_hidden_nodes = 100,
                    learning_rate = 0.1)

for i in range(len(train_imgs)):
    ANN.train(train_imgs[i], train_labels_one_hot[i])
 
synthesizer = Synthesizer(10, 28*28, 1, 100, 0.7, 100, digitsRandomizeFunction, ANN.run)
print synthesizer.synthesize(10)"""
