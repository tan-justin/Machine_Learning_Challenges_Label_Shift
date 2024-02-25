'''
Name: Justin Tan
Assignment: Adapt to Change
Date: Feb 25 2024
File: shift_adapt.py
'''

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from label_shift_adaptation import analyze_val_data, update_probs
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

'''
Type: Class
Name: ShiftAdapt
Purpose: Self-created "library" to contain methods for calculating accuracies and shifting labels
Parameters: None
-----------------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: __init__
Purpose: Preprocessing data and preparation of models
Parameters: train set, validation set, test set 1, test set 2, test set 3 (dataframes)
-----------------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: train
Purpose: Training all 6 models on the training set. Also prints out the feature importance required for extra credit
Parameters: train set, random forest classifier, 3nn classifier, 9nn classifier, gaussian classifier, baseline classifier x2
-----------------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: valid
Purpose: Get the accuracy of each classifier's prediction on the validation set, print graphs of truth label vs predicted label distribution
         for each classifier
Parameters: valid set, the 6 classifiers
-----------------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: test1
Purpose: Get the accuracy of each classifier's predictions on the test1 set, print graphs of truth label vs predicted label distribution
         for each classifier
Paramters: test1 set, the 6 classifiers
-----------------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: test2
Purpose: Get the accuracy of each classifier's predictions on the test2 set, print graphs of truth label vs predicted label distribution
         for each classifier
Parameters: test2 set, the 6 classifiers
-----------------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: test3
Purpose: Get the accuracy of each classifier's predictions on the test3 set, print graphs of truth label vs predicted label distribution
         for each classifier
Parameters: test3 set, the 6 classifiers
-----------------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: label_shift_1
Purpose: Perform label shift on test1 set and update the predictions. Prints the confusion matrix for each classifier and prints out
         the weights for each classifier
Parameters: test1 set, valid set, the classifiers excluding the baseline
-----------------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: label_shift_2
Purpose: Perform label shift on test2 set and update the predictions. Prints the confusion matrix for each classifier and prints out
         the weights for each classifier
Parameters: test2 set, valid set, the classifiers excluding the baseline
-----------------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: label_shift_3
Purpose: Perform label shift on test3 set and update the predictions. Prints the confusion matrix for each classifier and prints out
         the weights for each classifier
Parameters: test3 set, valid set, the classifiers excluding the baseline
-----------------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: plot_normalized_true_class_label_distribution_all_datasets
Purpose: Plot a graph showing the distribution of the truth labels for the valid set, test1 set, test2 set and test3 set
Paramters: valid set, test1 set, test2 set, test3 set
'''

class ShiftAdapt:

    def __init__(self, train, val, test1, test2, test3):
        self.data = train
        self.train_y = train['Severity']
        self.train_X = train.drop(columns = ['Severity'])

        self.val = val
        self.val_y = val['Severity']
        self.val_X = val.drop(columns = ['Severity'])


        self.test1_y = test1['Severity']
        self.test1_X = test1.drop(columns = ['Severity'])

        self.test2_y = test2['Severity']
        self.test2_X = test2.drop(columns = ['Severity'])

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

        feature_names = X.columns.tolist()
        feat_importance = self.rf.feature_importances_
        feature_importance_dict = {feature_names[i]: importance for i, importance in enumerate(feat_importance)}
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features:
            print(f"Feature: {feature}, Importance: {importance}")

    def valid(self):
        save_dir = 'valid_graphs'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        X = self.val_X.copy()
        y = self.val_y.copy()
        accuracy = {}
        for name, clf in self.list_classifiers.items():
            y_pred = clf.predict(X)
            accuracy[name] = round(accuracy_score(y, y_pred),2)
            cm = confusion_matrix(y, y_pred)
            total_instances = np.sum(cm)
            normalized_cm = cm / total_instances
            fig, ax = plt.subplots(figsize = (10, 6))
            labels = np.unique(np.concatenate((y, y_pred)))
            positions = np.arange(len(labels))
            ax.bar(positions - 0.2, normalized_cm.sum(axis=1), width=0.4, label='True Labels', color='b')
            ax.bar(positions + 0.2, normalized_cm.sum(axis=0), width=0.4, label='Predicted Labels', color='r')
            ax.set_xlabel('Class Label')
            ax.set_ylabel('Normalized Frequency')
            ax.set_title(f'Normalized True and Predicted Class Label Distribution ({name})')
            ax.set_xticks(positions)
            ax.set_xticklabels(labels)
            ax.legend()
            plt.tight_layout()
            filename = f'normalized_class_label_distribution_{name}_valid_set.png'
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath)
            plt.close()

        print('valid')
        
        return accuracy
    
    def test1(self):
        save_dir = 'test1_graphs'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        X = self.test1_X.copy()
        y = self.test1_y.copy()
        accuracy = {}
        for name, clf in self.list_classifiers.items():
            y_pred = clf.predict(X)
            accuracy[name] = round(accuracy_score(y, y_pred),2)
            cm = confusion_matrix(y, y_pred)
            total_instances = np.sum(cm)
            normalized_cm = cm / total_instances
            fig, ax = plt.subplots(figsize = (10, 6))
            labels = np.unique(np.concatenate((y, y_pred)))
            positions = np.arange(len(labels))
            ax.bar(positions - 0.2, normalized_cm.sum(axis=1), width=0.4, label='True Labels', color='b')
            ax.bar(positions + 0.2, normalized_cm.sum(axis=0), width=0.4, label='Predicted Labels', color='r')
            ax.set_xlabel('Class Label')
            ax.set_ylabel('Normalized Frequency')
            ax.set_title(f'Normalized True and Predicted Class Label Distribution ({name})')
            ax.set_xticks(positions)
            ax.set_xticklabels(labels)
            ax.legend()
            plt.tight_layout()
            filename = f'normalized_class_label_distribution_{name}_test1_set.png'
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath)
            plt.close()

        print('test1')

        return accuracy
    
    def test2(self):
        save_dir = 'test2_graphs'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        X = self.test2_X.copy()
        y = self.test2_y.copy()
        accuracy = {}
        for name, clf in self.list_classifiers.items():
            y_pred = clf.predict(X)
            accuracy[name] = round(accuracy_score(y, y_pred),2)
            cm = confusion_matrix(y, y_pred)
            total_instances = np.sum(cm)
            normalized_cm = cm / total_instances
            fig, ax = plt.subplots(figsize = (10, 6))
            labels = np.unique(np.concatenate((y, y_pred)))
            positions = np.arange(len(labels))
            ax.bar(positions - 0.2, normalized_cm.sum(axis=1), width=0.4, label='True Labels', color='b')
            ax.bar(positions + 0.2, normalized_cm.sum(axis=0), width=0.4, label='Predicted Labels', color='r')
            ax.set_xlabel('Class Label')
            ax.set_ylabel('Normalized Frequency')
            ax.set_title(f'Normalized True and Predicted Class Label Distribution ({name})')
            ax.set_xticks(positions)
            ax.set_xticklabels(labels)
            ax.legend()
            plt.tight_layout()
            filename = f'normalized_class_label_distribution_{name}_test2_set.png'
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath)
            plt.close()
        print('test2')
        
        return accuracy

    def test3(self):
        save_dir = 'test3_graphs'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        X = self.test3_X.copy()
        y = self.test3_y.copy()
        accuracy = {}
        for name, clf in self.list_classifiers.items():
            y_pred = clf.predict(X)
            accuracy[name] = round(accuracy_score(y, y_pred),2)
            cm = confusion_matrix(y, y_pred)
            total_instances = np.sum(cm)
            normalized_cm = cm / total_instances
            fig, ax = plt.subplots(figsize = (10, 6))
            labels = np.unique(np.concatenate((y, y_pred)))
            positions = np.arange(len(labels))
            ax.bar(positions - 0.2, normalized_cm.sum(axis=1), width=0.4, label='True Labels', color='b')
            ax.bar(positions + 0.2, normalized_cm.sum(axis=0), width=0.4, label='Predicted Labels', color='r')
            ax.set_xlabel('Class Label')
            ax.set_ylabel('Normalized Frequency')
            ax.set_title(f'Normalized True and Predicted Class Label Distribution ({name})')
            ax.set_xticks(positions)
            ax.set_xticklabels(labels)
            ax.legend()
            plt.tight_layout()
            filename = f'normalized_class_label_distribution_{name}_test3_set.png'
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath)
            plt.close()
        print('test3')

        return accuracy

    def label_shift_1(self):
        val_x = self.val_X
        val_y = self.val_y
        test_x = self.test1_X
        test_y = self.test1_y

        weights = {}
        accuracy = {}
        probability = {}
        class_labels = sorted(val_y.unique())

        list_classifiers = self.list_classifiers.copy()
        del list_classifiers['d_freq']
        del list_classifiers['d_strat']

        confusion_matrices = {}

        for name, clf in list_classifiers.items():
            val_pred = clf.predict(val_x)
            test_pred = clf.predict(test_x)
            test_prob = clf.predict_proba(test_x)
            weight = analyze_val_data(val_y, val_pred, test_pred)
            weights[name] = weight
            new_test_pred, new_test_prob = update_probs(class_labels, weight, test_pred, test_prob)
            accuracy[name] = round(accuracy_score(test_y, new_test_pred), 2)
            probability[name] = new_test_prob

            confusion_matrices[name] = confusion_matrix(test_y, new_test_pred)

        for name, matrix in confusion_matrices.items():
            print(f"Confusion matrix for classifier {name}:")
            print(matrix)
        print("Distribution of predictions on test set 1:")
        print("===========================================")
        for name, acc in accuracy.items():
            print(f"Classifier {name}: Accuracy = {acc}")
        print("BBSC adaptation weights:")
        print("========================")
        for item, weight in weights.items():
            print(f"Classifier {item}: Weight = {weight}")

        return weights, accuracy, probability
    
    def label_shift_2(self):
        val_x = self.val_X
        val_y = self.val_y
        test_x = self.test2_X
        test_y = self.test2_y

        weights = {}
        accuracy = {}
        probability = {}
        class_labels = sorted(val_y.unique())

        list_classifiers = self.list_classifiers.copy()
        del list_classifiers['d_freq']
        del list_classifiers['d_strat']

        confusion_matrices = {}

        for name, clf in list_classifiers.items():
            val_pred = clf.predict(val_x)
            test_pred = clf.predict(test_x)
            test_prob = clf.predict_proba(test_x)
            weight = analyze_val_data(val_y, val_pred, test_pred)
            weights[name] = weight
            new_test_pred, new_test_prob = update_probs(class_labels, weight, test_pred, test_prob)
            accuracy[name] = round(accuracy_score(test_y, new_test_pred), 2)
            probability[name] = new_test_prob
            confusion_matrices[name] = confusion_matrix(test_y, new_test_pred)
        
        for name, matrix in confusion_matrices.items():
            print(f"Confusion matrix for classifier {name}:")
            print(matrix)
        print("Distribution of predictions on test set 2:")
        print("===========================================")
        for name, acc in accuracy.items():
            print(f"Classifier {name}: Accuracy = {acc}")
        print("BBSC adaptation weights:")
        print("========================")
        for item, weight in weights.items():
            print(f"Classifier {item}: Weight = {weight}")

        return weights, accuracy, probability
    
    def label_shift_3(self):
        val_x = self.val_X
        val_y = self.val_y
        test_x = self.test3_X
        test_y = self.test3_y

        weights = {}
        accuracy = {}
        probability = {}
        class_labels = sorted(val_y.unique())

        list_classifiers = self.list_classifiers.copy()
        del list_classifiers['d_freq']
        del list_classifiers['d_strat']
        confusion_matrices = {}

        for name, clf in list_classifiers.items():
            val_pred = clf.predict(val_x)
            test_pred = clf.predict(test_x)
            test_prob = clf.predict_proba(test_x)
            weight = analyze_val_data(val_y, val_pred, test_pred)
            weights[name] = weight
            new_test_pred, new_test_prob = update_probs(class_labels, weight, test_pred, test_prob)
            accuracy[name] = round(accuracy_score(test_y, new_test_pred), 2)
            probability[name] = new_test_prob
            confusion_matrices[name] = confusion_matrix(test_y, new_test_pred)

        for name, matrix in confusion_matrices.items():
            print(f"Confusion matrix for classifier {name}:")
            print(matrix)
        print("Distribution of predictions on test set 3:")
        print("===========================================")
        for name, acc in accuracy.items():
            print(f"Classifier {name}: Accuracy = {acc}")
        print("BBSC adaptation weights:")
        print("========================")
        for item, weight in weights.items():
            print(f"Classifier {item}: Weight = {weight}")
   
        return weights, accuracy, probability
    
    def plot_normalized_true_class_label_distribution_all_datasets(self):
        save_dir = 'truth_distribution_graphs'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        class_labels = [1, 2, 3, 4]
        datasets = {'valid': self.val_y, 'test1': self.test1_y, 'test2': self.test2_y, 'test3': self.test3_y}
        dataset_counts = {}
        for dataset_name, y in datasets.items():
            dataset_counts[dataset_name] = [np.sum(y == label) / len(y) for label in class_labels]

        fig, ax = plt.subplots(figsize=(10, 6))
        width = 0.2
        positions = np.arange(len(class_labels))

        for i, (dataset_name, counts) in enumerate(dataset_counts.items()):
            ax.bar(positions + (i - 1.5) * width, counts, width = width, label = dataset_name)
        ax.set_xlabel('Class Label')
        ax.set_ylabel('Normalized Frequency')
        ax.set_title('Normalized True Class Label Distribution')
        ax.set_xticks(positions)
        ax.set_xticklabels(class_labels)
        ax.legend()
        plt.tight_layout()
        filename = f'normalized_truth_label_distribution_set.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.close()

