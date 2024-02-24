import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from label_shift_adaptation import analyze_val_data, update_probs

class ShiftAdapt:

    def __init__(self, train, val, test1, test2, test3):
        self.data = train
        self.train_y = train['Severity']
        self.train_X = train.drop(columns = ['Severity'])

        self.val = val
        self.val_y = val['Severity']
        self.val_X = val.drop(columns = ['Severity'])

        #self.test1 = test1
        self.test1_y = test1['Severity']
        self.test1_X = test1.drop(columns = ['Severity'])

        #self.test2 = test2
        self.test2_y = test2['Severity']
        self.test2_X = test2.drop(columns = ['Severity'])

        #self.test3 = test3
        self.test3_y = test3['Severity']
        self.test3_X = test3.drop(columns = ['Severity'])

        self.rf = RandomForestClassifier(random_state = 0)
        self.nn3 = KNeighborsClassifier(n_neighbors = 3)
        self.nn9 = KNeighborsClassifier(n_neighbors = 9)
        self.gpc = GaussianProcessClassifier(random_state = 0)
        self.d_freq = DummyClassifier(strategy = 'most_frequent', random_state = 0)
        self.strat = DummyClassifier(strategy = 'stratified', random_state = 0)

        self.list_classifiers = {'rf':self.rf, '3nn':self.nn3, '9nn':self.nn9, 'gpc':self.gpc, 'd_freq':self.d_freq, 'd_strat':self.strat}

    def train(self):
        X = self.train_X.copy()
        y = self.train_y.copy()
        self.rf.fit(X,y)
        self.nn3.fit(X,y)
        self.nn9.fit(X,y)
        self.gpc.fit(X, y)
        self.d_freq.fit(X,y)
        self.strat.fit(X,y)

        print('Successfully trained all models')

    def valid(self):
        X = self.val_X.copy()
        y = self.val_y.copy()
        accuracy = {}
        for name, clf in self.list_classifiers.items():
            y_pred = clf.predict(X)
            accuracy[name] = round(accuracy_score(y, y_pred),2)
        print('valid:')
        print(accuracy)
        return accuracy
    
    def test1(self):
        X = self.test1_X.copy()
        y = self.test1_y.copy()
        accuracy = {}
        for name, clf in self.list_classifiers.items():
            y_pred = clf.predict(X)
            accuracy[name] = round(accuracy_score(y, y_pred),2)
        print('test1:')
        print(accuracy)
        return accuracy
    
    def test2(self):
        X = self.test2_X.copy()
        y = self.test2_y.copy()
        accuracy = {}
        for name, clf in self.list_classifiers.items():
            y_pred = clf.predict(X)
            accuracy[name] = round(accuracy_score(y, y_pred),2)
        print('test2:')
        print(accuracy)
        return accuracy

    def test3(self):
        X = self.test3_X.copy()
        y = self.test3_y.copy()
        accuracy = {}
        for name, clf in self.list_classifiers.items():
            y_pred = clf.predict(X)
            accuracy[name] = round(accuracy_score(y, y_pred),2)
        print('test3:')
        print(accuracy)
        return accuracy

    def label_shift_1(self):
        val_x = self.val_X
        val_y = self.val_y
        test_x = self.test1_X
        test_y = self.test1_y

        weights = {}
        accuracy = {}
        probability = {}
        class_labels = val_y.unique()
        print(class_labels)

        list_classifiers = self.list_classifiers.copy()
        del list_classifiers['d_freq']
        del list_classifiers['d_strat']

        for name, clf in list_classifiers.items():
            val_pred = clf.predict(val_x)
            test_pred = clf.predict(test_x)
            test_prob = clf.predict_proba(test_x)
            weight = analyze_val_data(val_y, val_pred, test_pred)
            weights[name] = weight
            new_test_pred, new_test_prob = update_probs(class_labels, weight, test_pred, test_prob)
            accuracy[name] = round(accuracy_score(test_y, new_test_pred), 2)
            probability[name] = new_test_prob

        print('weights:')
        print(weights)
        
        print('accuracy:')
        print(accuracy)

        print('probability:')
        print(probability)

        return weights, accuracy, probability
    
    def label_shift_2(self):
        val_x = self.val_X
        val_y = self.val_y
        test_x = self.test2_X
        test_y = self.test2_y

        weights = {}
        accuracy = {}
        probability = {}
        class_labels = val_y.unique()
        print(class_labels)

        list_classifiers = self.list_classifiers.copy()
        del list_classifiers['d_freq']
        del list_classifiers['d_strat']

        for name, clf in list_classifiers.items():
            val_pred = clf.predict(val_x)
            test_pred = clf.predict(test_x)
            test_prob = clf.predict_proba(test_x)
            weight = analyze_val_data(val_y, val_pred, test_pred)
            weights[name] = weight
            new_test_pred, new_test_prob = update_probs(class_labels, weight, test_pred, test_prob)
            accuracy[name] = round(accuracy_score(test_y, new_test_pred), 2)
            probability[name] = new_test_prob

        print('weights:')
        print(weights)
        
        print('accuracy:')
        print(accuracy)

        print('probability:')
        print(probability)

        return weights, accuracy, probability
    
    def label_shift_3(self):
        val_x = self.val_X
        val_y = self.val_y
        test_x = self.test3_X
        test_y = self.test3_y

        weights = {}
        accuracy = {}
        probability = {}
        class_labels = val_y.unique()
        print(class_labels)

        list_classifiers = self.list_classifiers.copy()
        del list_classifiers['d_freq']
        del list_classifiers['d_strat']

        for name, clf in list_classifiers.items():
            val_pred = clf.predict(val_x)
            test_pred = clf.predict(test_x)
            test_prob = clf.predict_proba(test_x)
            weight = analyze_val_data(val_y, val_pred, test_pred)
            weights[name] = weight
            new_test_pred, new_test_prob = update_probs(class_labels, weight, test_pred, test_prob)
            accuracy[name] = round(accuracy_score(test_y, new_test_pred), 2)
            probability[name] = new_test_prob

        print('weights:')
        print(weights)
        
        print('accuracy:')
        print(accuracy)

        print('probability:')
        print(probability)

        return weights, accuracy, probability

