#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import os
from data_preparation.prepare_data import create_train_and_test_sets, import_sets

def import_data_for_classification(spreadsheets_dir, data_dir, features = 'all'):
    """Imports and parses spreadsheets as data structures for classification
    and save them as CSV files.

    Args:
        spreadsheets_dir (String): Folder where spreadsheets
        are located.
        data_dir (String): Folder where parsed data is saved.

    Returns:
        Dataframes: Training and test samples (See import_sets in prepare_data.py)
    """

    # Spreadsheets headers
    text_column = 'Paragraph'   
    classes_columns = ['No categories identified.',
                       'CF – Contribution flow',
                       'CT – Choose a task',
                       'TC – Talk to the community',
                       'BW – Build local workspace',
                       'DC – Deal with the code',
                       'SC – Submit the changes']

    # Label for a new column header that will merge
    # classes_columns into a single column
    label_column = 'Label'

    # Filepaths where the train and test sets are saved
    train_filepath = os.path.join(data_dir, 'train.csv')
    # test_filepath = os.path.join(data_dir, 'test.csv')
    test_filepath = os.path.join(data_dir, 'test_without_duplicates.csv') # Using to respond reviewer

    if not os.path.exists(train_filepath) or not os.path.exists(test_filepath):
        print('Generating train and test sets...')
        create_train_and_test_sets(spreadsheets_dir, text_column, 
                                   classes_columns, train_filepath, test_filepath,
                                   label_column)

    return import_sets(train_filepath, test_filepath, text_column, label_column, features=features)

def import_data_for_prediction(spreadsheets_dir, data_dir):
    """Imports and parses spreadsheets as data structures for prediction.
    Notice that such spreadsheets will not be used to train a classifier,
    but to predict the performance of it on unseen data.

    Args:
        spreadsheets_dir (String): Folder where spreadsheets
        are located.
        data_dir (String): Folder where parsed data is saved.
    Returns:
        Dataframes: Training and test samples (See import_sets in prepare_data.py)
    """

    # Spreadsheets headers
    text_column = 'Paragraph'   
    classes_columns = ['No categories identified.',
                       'CF – Contribution flow',
                       'CT – Choose a task',
                       'TC – Talk to the community',
                       'BW – Build local workspace',
                       'DC – Deal with the code',
                       'SC – Submit the changes']

    # Label for a new column header that will merge
    # classes_columns into a single column
    label_column = 'Prediction'

    # Filepaths where the train and test sets are saved
    # Notice that we are using data from the folder 'for-prediction'
    # These sets are merged later in main.py
    train_filepath = os.path.join(data_dir, 'train_predict.csv')
    test_filepath = os.path.join(data_dir, 'test_predict.csv')

    if not os.path.exists(train_filepath) or not os.path.exists(test_filepath):
        create_train_and_test_sets(spreadsheets_dir, text_column, 
                                   classes_columns, train_filepath, test_filepath,
                                   label_column)

    return import_sets(train_filepath, test_filepath, text_column, label_column, True)