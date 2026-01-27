#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import os
import json
import matplotlib.pyplot as plot
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve

import numpy as np
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from imblearn.over_sampling import SMOTE

def export_classification_report(model, X_test, y_test, results_dir):
    classes = ['No categories identified.',
               'CF - Contribution flow',
               'CT – Choose a task',
               'TC – Talk to the community',
               'BW – Build local workspace',
               'DC – Deal with the code',
               'SC – Submit the changes']

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)

    report_filepath = os.path.join(results_dir, 'classification_report_duplicates_removed.json')

    with open(report_filepath, 'w') as report_file:
        json.dump(report, report_file, indent=4)


def export_confusion_matrix(model, X_test, y_test):
    classes = ['No categories identified.',
                'CF - Contribution flow',
               'CT – Choose a task',
               'TC – Talk to the community',
               'BW – Build local workspace',
               'DC – Deal with the code',
               'SC – Submit the changes']

    labels = [category[:2] for category in classes]
    plot_confusion_matrix(model, X_test, y_test, display_labels=labels, cmap=plot.cm.Blues)
    plot.show()


def export_learning_curve(classifier, strategy, oversample, X_train, y_train):
    plt.figure()

    if strategy == 'one_vs_rest':
        estimator = OneVsRestClassifier(classifier)
    if strategy == 'one_vs_one':
        estimator = OneVsOneClassifier(classifier)

    if oversample:
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)

    train_sizes, train_scores, test_scores = learning_curve(estimator, X_train, y_train, cv=10, train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, color="blue", marker="o", markersize=5, label="Training Accuracy")
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

    plt.xlabel("Trainining Data Size")
    plt.ylabel("Model Accuracy")
    plt.title('Learning Curve')
    plt.grid()
    plt.legend(loc="best")
    plt.show()