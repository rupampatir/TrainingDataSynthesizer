import random as random
from classifiers.income.income_classifier import IncomeClassifier
from classifiers.digits.digits import NeuralNetwork
from synthesizer import Synthesizer
import pandas as pd
import numpy as np
import pickle
import csv
import matplotlib.pyplot as plt
import math as math

############################
###### INCOME DATASET ######
############################

"""
def incomeRandomizeFunction(k, x, c):
    x_age = random.randint(0, 100)
    x_education = random.randint(0, 16)
    x_marital_status = random.randint(0, 1)
    x_sex = random.randint(0, 1)
    x_gain = random.randint(0, 100000)
    x_loss = random.randint(0, 100000)
    x_hpw = random.randint(0, 100)
    x_temp = [x_age, x_education, x_marital_status, x_sex, x_gain, x_loss, x_hpw]

    if len(x) == 0:
        return x_temp

    selected_features = random.sample(range(len(x_temp)), k)

    for argfeature in selected_features:
        x[argfeature] = x_temp[argfeature]

    return x


print "Loading Income Classifier Data\n"
dataset = pd.read_csv("classifiers/income/adult.csv")
print "Intializing Classifier\n"
income_classifier = IncomeClassifier(dataset)
print "Intializing Training Data Synthesis\n"

synthesizer = Synthesizer(
    2, 7, 1, 100, 0.9, 5, incomeRandomizeFunction, income_classifier.predict_probabilities)
    #c,kmax,kmin,iter_max,conf_min,rej_max,randomizeFunction,predictProb
synthesizer.synthesize(10) #number of records
print "Testing probabilities..."
print "Class 0 means less than 50K. 1 means greater than 50K"
for c in range(2):
    dataset = list(csv.reader(open("training_class_"+str(c)+".csv")))
    print "\n\nProb dist for class" + str(c)
    for record in dataset:
        print income_classifier.predict_probabilities(record)
"""
###########################
###### DIGITS DATASET #####
###########################

print "Loading MNIST Classifier Data\n"
with open("classifiers/digits/pickled_mnist.pkl", "rb") as fh:
    data = pickle.load(fh)
train_imgs = data[0]
test_imgs = data[1]
train_labels = data[2]
test_labels = data[3]
train_labels_one_hot = data[4]
test_labels_one_hot = data[5]
image_size = 28  # width and length
no_of_different_labels = 10  # i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size

print "Initialising NN Classifier\n"

ANN = NeuralNetwork(no_of_in_nodes=image_pixels,
                    no_of_out_nodes=10,
                    no_of_hidden_nodes=100,
                    learning_rate=0.1)

for i in range(len(train_imgs)):
    ANN.train(train_imgs[i], train_labels_one_hot[i])


### Synthesis Code ###
print "Intializing Training Data Synthesis\n"

init_x = {}
for digit in range(0, 10):
    init_x[digit] = [i for i, x in enumerate(test_labels) if x == digit][:10]

def digitsRandomizeFunction(k, x, c):
    fac = 0.99 / 255

    if len(x) == 0:
        x = np.asfarray(
            test_imgs[init_x[c][random.randint(0, 9)]]) * fac + 0.01

    gray_features = [i for i, feature in enumerate(x) if feature > np.min(x)]

    # There may be less gray pixels i.e. less than k. For this we check how many
    # 'extra features' there are and repeat the process of manipulating feature
    # values to compensate this scarcity

    extra_features = 0
    selected_gray_features = gray_features
    if len(gray_features) >= k:
        selected_gray_features = random.sample(gray_features, k)
    else:
        extra_features = k - len(gray_features)
    
    #we define a stride i.e. the distance from the selected pixel that can be altered
    stride = 1

    for grayfeature in selected_gray_features:
        # choose a nearby pixel. stride pixels away from the selected gray feature
        row_number = int(grayfeature/28)
        column_number = grayfeature%28
        new_row = random.randint(max(row_number - stride, 0), min(row_number +  stride, 27))
        new_col = random.randint(max(column_number - stride, 0), min(column_number +  stride, 27))
        indice_to_change = new_row * 28 + new_col
        x[indice_to_change] = random.randint(0, 255) * fac + 0.01

    #Repeat the process for the extra features
    for extra_feature in range(extra_features):
        grayfeature = selected_gray_features[random.randint(0, len(selected_gray_features) - 1)]
        row_number = int(grayfeature/28)
        column_number = grayfeature%28
        new_row = random.randint(max(row_number - stride, 0), min(row_number +  stride, 27))
        new_col = random.randint(max(column_number - stride, 0), min(column_number +  stride, 27))
        indice_to_change = new_row * 28 + new_col
        x[indice_to_change] = random.randint(0, 255) * fac + 0.01

    return x

def digitsPredictProb(x):
    output_vector = np.array(ANN.run(x))
    return output_vector/sum(output_vector)


synthesizer = Synthesizer(10, 28*28, 1, 200, 0.95, 200,
                          digitsRandomizeFunction, digitsPredictProb)
                          #c, kmax, kmin, iter_max, conf_min, rej_max,
                          #randomizeFunction, predictProb

print synthesizer.synthesize(10)

print "Testing probabilities..."
for c in range(10):
    record = [float(i) for i in np.array(list(csv.reader(open("training_class_"+str(c)+".csv")))[0])]
    print "\n\nProb dist for class" + str(c)
    img = np.array(record).reshape((28,28))
    plt.imshow(img, cmap="Greys")
    plt.show()
    print digitsPredictProb(record)
