import random as random
import numpy as np
import csv

class Synthesizer:
    def __init__(self,
                 c,
                 kmax,
                 kmin,
                 iter_max,
                 conf_min,
                 rej_max,
                 randomizeFunction,
                 predictProb,
                 ):
        self.c = c
        self.kmax = kmax
        self.kmin = kmin
        self.iter_max = iter_max
        self.conf_min = conf_min
        self.rej_max = rej_max,
        self.randomizeFunction = randomizeFunction
        self.predictProb = predictProb

    def randomize(self, k, x):

        x_temp = self.randomizeFunction()

        if len(x) == 0:
            return x_temp

        selected_features = random.sample(range(len(x_temp)), k)

        for argfeature in selected_features:
            x[argfeature] = x_temp[argfeature]

        return x

    def synthesizeRecord(self, c, k_max, k_min, iter_max, conf_min, rej_max):
        y_current = 0
        x_current = []
        j = 0
        k = k_max
        x = self.randomize(k, x_current)

        for iteration in range(iter_max):
            y = self.predictProb(x)
            if y[c] >= y_current:
                if y[c] > conf_min and c == np.argmax(y):
                    if (random.random() < y[c]):
                        return x
                x_current = x
                y_current = y[c]
                j = 0
            else:
                j += 1
                if j > rej_max:
                    k = np.max([k_min, k/2])
                    j = 0
            x = self.randomize(k, x_current)

        return False  # Failed

    def synthesize(self, total_number_of_records):

        for c in range(self.c):
            print 'Synthesizing data for class ' + str(c)
            current_number_of_records = 0
            with open('training_class_'+str(c)+'.csv', 'a') as csvFile:
                writer = csv.writer(csvFile)
                while current_number_of_records <= total_number_of_records:
                    record = self.synthesizeRecord(
                        c, self.kmax, self.kmin, self.iter_max, self.conf_min, self.rej_max)
                    if record:
                        current_number_of_records += 1
                        # synthesized_training_data.append(record) #c, kmax, kmin, iter_max, conf_min, rej_max
                        writer.writerow(record)

        print '\n\nSynthesis Completed'
