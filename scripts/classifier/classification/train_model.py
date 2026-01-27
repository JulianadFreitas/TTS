#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import os
import pandas

# Classifier training
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from imblearn.over_sampling import SMOTE

# Cross-validation
from sklearn.model_selection import cross_validate

def train_classifier(classifier, strategy, oversample, X_train, y_train):
    """Computes a multiclass classification.

    Args:
        classifier: An instance of a scikit-learn classifier.
        strategy: A string defining which strategies will be used for training.
        oversample: A boolean variable that defines if SMOTE oversampling should be applied or not.
        X_train: A matrix containing features for training.
        X_test: A matrix containing features for testing.
        y_train: A column containing labels for training.
        y_test: A column containing labels for testing.

    Returns:
        A classification model and its performance report
    """

    if strategy == 'one_vs_rest':
        model = OneVsRestClassifier(classifier)
    if strategy == 'one_vs_one':
        model = OneVsOneClassifier(classifier)

    if oversample:
        X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)
        model.fit(X_resampled, y_resampled)
    else:
        model.fit(X_train, y_train)

    return model

def features_cross_validation(classifier, strategy, oversample, X_train, y_train, feature_names, results_dir):
    print("Getting feature coefficients from LinearSVC")
    folds = None

    if strategy == 'one_vs_rest':
        model = OneVsRestClassifier(classifier)
    if strategy == 'one_vs_one':
        model = OneVsOneClassifier(classifier)

    if oversample:
        X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)
        folds = cross_validate(model, X_resampled, y_resampled, cv = 10, return_estimator=True)
    else:
        folds = cross_validate(model, X_train, y_train, cv = 10, return_estimator=True)
    
    folds_concat = pandas.DataFrame()

    for estimator in folds['estimator']:
        weights = pandas.DataFrame(estimator.coef_, index=estimator.classes_.tolist(), columns=feature_names)
        folds_concat = pandas.concat((folds_concat, weights))
    
    ordered_folds = folds_concat.sort_index()
    ordered_folds.to_csv(os.path.join(results_dir, 'features_weights.csv'))
    by_category = folds_concat.groupby(folds_concat.index)
    weights_mean = by_category.mean()
    best_features_per_class = weights_mean.apply(lambda s, n: pandas.Series(s.nlargest(n).index), axis=1, n=5)
    weights_mean.to_csv(os.path.join(results_dir, 'features_weights_mean.csv'))
    best_features_per_class.to_csv(os.path.join(results_dir, 'best_features_per_class.csv'))