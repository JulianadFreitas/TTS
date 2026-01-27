#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import pandas
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split

from .generate_features import create_statistic_features, create_heuristic_features
from .transform_data import transform_spreadsheets_in_dataframe
from .preprocess_text import text_preprocessing
from .select_features import select_features

def create_train_and_test_sets(spreadsheets_dir, text_column, classes_columns,
                               train_filepath, test_filepath, label_column):
    """Creates the train and test sets based on the spreadsheets from the 
        qualitative analysis.

    Args:
        spreadsheets_dir (String): Represents the path to the spreadsheets folder
        text_column (String): Represents the column containing the paragraphs
            in each spreadsheet
        classes_columns (List of strings): Represent the columns representing
            classes in each spreadsheet
        train_filepath (String): Represents the filepath where the train set should be
            saved as a CSV file
        test_filepath (String): Represents the filepath where the test set should be
            saved as a CSV file
        label_column (String): Represents the name given to a new column that
            will be used to store the label of each paragraph.
    """
    dataframe = transform_spreadsheets_in_dataframe(spreadsheets_dir,
                                                    text_column,
                                                    classes_columns,
                                                    label_column)

    X = dataframe[text_column]
    y = dataframe[label_column]
    
    if label_column == 'Prediction':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

    train_data = pandas.concat([X_train, y_train], axis=1)
    test_data = pandas.concat([X_test, y_test], axis=1)

    train_data.to_csv(train_filepath, index=False, encoding='utf-8-sig')
    test_data.to_csv(test_filepath, index=False, encoding='utf-8-sig')

def import_sets(train_filepath, test_filepath, text_column, label_column, is_predict = False, features='all'):
    """Imports train and test sets and applies the text preprocessing techniques when necessary
    
    Args:
        text_column (String): Represents the column containing the paragraphs
            in each spreadsheet
        train_filepath (String): Represents the filepath where the train instances are
            saved as a CSV file
        test_filepath (String): Represents the filepath where the test instances are
            saved as a CSV file
        label_column (String): Represents the name given to a new column that
            will be used to store the label of each paragraph.
        is_predict (Bool, optional): If the purpose of
            feature selection is prediction, it loads the
            feature selector used during training to avoid
            overfitting. Defaults to False.
    """
    print("Importing training and test sets.")
    train_data = pandas.read_csv(train_filepath)
    train_text_column = train_data[text_column]
    X_train, y_train = train_data[text_column], train_data[label_column]

    test_data = pandas.read_csv(test_filepath)
    test_text_column = test_data[text_column]
    X_test, y_test = test_data[text_column], test_data[label_column]

    print("Applying preprocessing techniques on paragraphs column.")
    preprocessing_techniques = ['remove-stopwords', 'remove-punctuations', 'lemmatization']
    X_train = text_preprocessing(X_train, preprocessing_techniques)
    X_test = text_preprocessing(X_test, preprocessing_techniques)

    print("Converting paragraphs into statistic features.")
    train_statistic_features, test_statistic_features = create_statistic_features(X_train, X_test, is_predict)

    print("Converting paragraphs into heuristic features.")
    train_heuristic_features, test_heuristic_features = create_heuristic_features(X_train, X_test)

    if features == 'all':
        X_train = pandas.concat([train_statistic_features, train_heuristic_features], axis=1)
        X_test = pandas.concat([test_statistic_features, test_heuristic_features], axis=1)
    elif features == 'heuristic':
        X_train = train_heuristic_features
        X_test = test_heuristic_features
    elif features == 'statistic':
        X_train = train_statistic_features
        X_test = test_statistic_features
    else:
        print('The type of features you aim to use does not exist.')
        raise ValueError

    print("Selecting features with SelectPercentile (chi2).")
    X_train, X_test, selected_feature_names = select_features(X_train, y_train, X_test, is_predict)

    return X_train, y_train, X_test, y_test, train_text_column, test_text_column, selected_feature_names
