#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import pickle
from sklearn.feature_selection import SelectPercentile, chi2

def select_features(X_train, y_train, X_test, is_predict = False):
    """Selects the best features in a classification problem

    Args:
        X_train (Dataframe): Training features
        y_train (Series): Training labels
        X_test (Dataframe): Test features
        is_predict (Bool, optional): If the purpose of
            feature selection is prediction, it loads the
            feature selector used during training to avoid
            overfitting. Defaults to False.
    Returns:
        Dataframe, Dataframe: Training and test features selected
            using SelectPercentile (chi-square)
    """

    if is_predict:
        selector = pickle.load(open('feature_selector.sav', 'rb'))
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)           
    else:
        selector = SelectPercentile(chi2, percentile=15)
        selector.fit(X_train, y_train)

        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)

        pickle.dump(selector, open('feature_selector.sav', 'wb'))

    return X_train, X_test, selector.get_feature_names_out()
