#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

# General modules
import csv
import os
import random
import shutil
import pandas
import pickle
import numpy as np
from scipy.sparse import vstack

# Final estimator 
from sklearn.svm import LinearSVC

# Data preparation
from data_preparation.import_data import import_data_for_classification, import_data_for_prediction

# Model selection
from model_selection.evaluate_estimators import evaluate_estimators_performance
from sklearn.ensemble import RandomForestClassifier as RClf

# Classification
from classification.train_model import train_classifier, features_cross_validation
from classification.explore_model import export_classification_report
from classification.explore_model import export_confusion_matrix
from classification.explore_model import export_learning_curve

def find_best_estimator(X_train, y_train, results_dir):
    """Tests a list of pre-defined algorithms with the 
    training samples provided in order to find
    the one that best fits the given problem.

    Args:
        X_train (Dataframe): Training features
        y_train (Series): Training labels
        results_dir (String): Folder were the
        performance results are supposed to be
        stored.
    """

    # Algorithms implemented:
    # 'svc' for Support Vector 
    # 'mnb' for Multinomial Naive Bayes
    # 'knn' for K-nearest Neighbors
    # 'lr' for Logistic Regression
    # 'rf' for Random Forest
    # 'dmr' for Dummy Classifier (Random)
    # 'dmf' for Dummy Classifier (Always the most frequent)
    classifiers = ['dmr', 'svc', 'mnb', 'knn', 'lr', 'rf', 'dmf'] # 

    # Strategies available:
    # 'ovr' for OneVsRest
    # 'ovo' for OneVsOne
    strategies = ['ovr', 'ovo']

    # Oversampling:
    # True to apply SMOTE
    # False to not apply SMOTE
    oversample = [True, False]

    evaluate_estimators_performance(classifiers, strategies, oversample,
                                    X_train, y_train, results_dir)

def evaluate_final_estimator_on_unseen_data(X_train, y_train, X_test, y_test, results_dir):
    """After identifying the algorithm that provides the best
    performance (See method find_best_estimator), this method
    evaluates such algorithm on unseen data (test instances), 
    providing a classification report, learning curves and 
    a confusing matrix.

    Args:
        X_train (Dataframe): Training features
        y_train (Series): Training labels
        X_test (Dataframe): Test features
        y_test (Series): Test labels
        results_dir (String): Folder where results
        should be saved.
    """

    # Based on the current tests, LinearSVC with the following arguments
    # is the estimator that provides the best performance for the training instances.
    selected_classifier = LinearSVC(tol=0.001, C=1, max_iter=500)

    training_args = {
        'classifier': selected_classifier,
        'strategy': 'one_vs_rest',
        'oversample': False,
        'X_train': X_train,
        'y_train': y_train
    }

    model = train_classifier(**training_args)

    export_classification_report(model, X_test, y_test, results_dir)
    # export_confusion_matrix(model, X_test, y_test)
    # export_learning_curve(**training_args)

def evaluate_usefulness_of_features(X_train, y_train, selected_feature_names, results_dir):

    # Based on the current tests, LinearSVC with the following arguments
    # is the estimator that provides the best performance for the training instances.
    selected_classifier = LinearSVC(tol=0.001, C=1, max_iter=500)

    training_args = {
        'classifier': selected_classifier,
        'strategy': 'one_vs_rest',
        'oversample': False,
        'X_train': X_train,
        'y_train': y_train
    }

    features_cross_validation(**training_args, feature_names=selected_feature_names, results_dir=results_dir)

def train_final_estimator(X_train, y_train, X_test, y_test):
    """After identifying the algorithm that provides the best
    performance (See method find_best_estimator), this method
    trains such estimator using all train and test instances, 
    and dumps a final model for prediction.

    Args:
        X_train (Dataframe): Training features
        y_train (Series): Training labels
        X_test (Dataframe): Test features
        y_test (Series): Test labels
    """
    selected_classifier = LinearSVC(tol=0.001, C=1.5, max_iter=500)

    # Merges training and test samples/labels
    X_train = vstack((X_train, X_test))
    y_train = pandas.concat([y_train, y_test])

    training_args = {
        'classifier': selected_classifier,
        'strategy': 'one_vs_rest',
        'oversample': False,
        'X_train': X_train,
        'y_train': y_train
    }

    model = train_classifier(**training_args)

    # Dumps the model to a file for future predictions.
    pickle.dump(model, open('final_estimator.sav', 'wb'))

def predict_survey_spreadsheets(spreadsheets_dir, predict_spreadsheets_dir, results_dir, n_samples):
    """Using the final version of the best estimator (See train_final_estimator), 
    this method is used to predict the classes of new data samples.

    This method is used in our survey where participants evaluate
    if our classifications make sense or not.

    Args:
        spreadsheets_dir (String): Folder where spreadsheets are located.
        Notice that such spreadsheets should not be used to train the
        classifier in the previous steps. In other works, they should
        be unknown to the final estimator.
        predict_spreadsheets_dir (String): Folder where a copy
        of such spreadsheets will be saved. 
        results_dir (String): Folder where the predictions will
        be saved.
        n_samples (Integer): Number of spreadsheets to be predicted.
    """

    # Finds a spreadsheet in the spreadsheet folder that is not
    # being used, and copies it to a new folder of spreadsheets
    # to be predicted. The outer loop stops when a number of
    # n_samples is satisfied.

    for i in range(0, n_samples):
        while True:
            filename = random.choice(os.listdir(spreadsheets_dir))
            filepath = os.path.join(spreadsheets_dir, filename)
            copy_filepath = os.path.join(predict_spreadsheets_dir, filename)

            if os.path.isfile(filepath) and not os.path.exists(copy_filepath):
                shutil.copyfile(filepath, copy_filepath)
                break

    # We use the same method using during the training process of an estimator
    # to parse the data from the spreadsheets to be predicted.
    X_train, _, X_test, _, train_text_column, test_text_colum, _ = import_data_for_prediction(predict_spreadsheets_dir, data_dir)

    # Merge training and test samples. Notice that this data
    # will not be used for training but only for prediction.
    # We just need to merge "training and test samples" because
    # we use the method import_data_for_prediction to parse the
    # data, which is also used in training.
    X_predict = vstack((X_train, X_test))
    text_column = pandas.concat([train_text_column, test_text_colum], ignore_index = True)

    # Loads the final estimator (See predict_using_final_estimator).
    model = pickle.load(open('final_estimator.sav', 'rb'))

    # Using the final estimator, predicts the classes for the unknown
    # spreadsheets.
    y_predict = model.predict(X_predict)

    # Saves predictions in a CSV file.
    with open(os.path.join(results_dir, 'predictions.csv'), 'w') as predictions_file:
        classes = ['No categories identified.',
                    'CF - Contribution flow',
                    'CT - Choose a task',
                    'TC - Talk to the community',
                    'BW - Build local workspace',
                    'DC - Deal with the code',
                    'SC - Submit the changes']

        for predicted_class in classes:
            predictions_file.write('Paragraph, Predicted Class\n')

            instances = [i for i, item in enumerate(y_predict) if item.startswith(predicted_class[:2])]

            for i in range(0, len(instances)):
                if len(text_column[instances[i]]) > 100:
                    predictions_file.write("\"%s\", %s\n" % (text_column[instances[i]], y_predict[instances[i]]))

            predictions_file.write('\n')

def predict_analysis_spreadsheets(spreadsheets_dir, predict_spreadsheets_dir, results_dir, n_samples):
    """Using the final version of the best estimator (See train_final_estimator), 
    this method is used to predict the classes of new data samples.

    This method is used in our survey where participants evaluate
    if our classifications make sense or not.

    Args:
        spreadsheets_dir (String): Folder where spreadsheets are located.
        Notice that such spreadsheets should not be used to train the
        classifier in the previous steps. In other works, they should
        be unknown to the final estimator.
        predict_spreadsheets_dir (String): Folder where a copy
        of such spreadsheets will be saved. 
        results_dir (String): Folder where the predictions will
        be saved.
        n_samples (Integer): Number of spreadsheets to be predicted.
    """

    # Finds a spreadsheet in the spreadsheet folder that is not
    # being used, and copies it to a new folder of spreadsheets
    # to be predicted. The outer loop stops when a number of
    # n_samples is satisfied.

    for i in range(0, n_samples):
        while True:
            filename = random.choice(os.listdir(spreadsheets_dir))
            filepath = os.path.join(spreadsheets_dir, filename)
            copy_filepath = os.path.join(predict_spreadsheets_dir, filename)

            if os.path.isfile(filepath) and not os.path.exists(copy_filepath):
                shutil.copyfile(filepath, copy_filepath)
                break

    # We use the same method using during the training process of an estimator
    # to parse the data from the spreadsheets to be predicted.
    X_train, y_train, X_test, y_test, train_text_column, test_text_column, _ = import_data_for_prediction(predict_spreadsheets_dir, data_dir)

    # Merge training and test samples. Notice that this data
    # will not be used for training but only for prediction.
    # We just need to merge "training and test samples" because
    # we use the method import_data_for_prediction to parse the
    # data, which is also used in training.
    X_predict = vstack((X_train, X_test))
    projects_filename = pandas.concat([y_train, y_test], ignore_index = True)  # Just used to get the projects' filename
    text_column = pandas.concat([train_text_column, test_text_column], ignore_index = True)
    print(y_train)

    # Loads the final estimator (See predict_using_final_estimator).
    model = pickle.load(open('final_estimator.sav', 'rb'))

    # Using the final estimator, predicts the classes for the unknown
    # spreadsheets.
    y_predict = model.predict(X_predict)

    # Saves predictions in a CSV file.
    with open(os.path.join(results_dir, 'predictions.csv'), 'w', newline='', encoding="utf-8") as predictions_file:
        writer = csv.writer(predictions_file)
        writer.writerow(['Paragraph', 'Class', 'Project'])

        for i in range(len(y_predict)):
            writer.writerow([text_column[i], y_predict[i], projects_filename[i]])

if __name__ == '__main__':
    # Folders used during the whole process:
    # repository/scripts/classifier/
    classifier_dir = os.getcwd()
    # repository/scripts/
    scripts_dir = os.path.dirname(classifier_dir)
    # repository/
    repository_dir = os.path.dirname(scripts_dir) 
    # repository/data/
    data_dir = os.path.join(repository_dir, 'data')
    # repository/results/
    results_dir = os.path.join(repository_dir, 'results') 
    # repository/data/documentation/spreadsheets/
    spreadsheets_dir = os.path.join(data_dir, 'documentation', 'spreadsheets')
    # repository/data/documentation/spreadsheets/valid
    training_spreadsheets_dir = os.path.join(spreadsheets_dir, 'for-training')
    # repository/data/documentation/spreadsheets/for-prediction
    survey_spreadsheets_dir = os.path.join(spreadsheets_dir, 'for-survey')

    ###########
    # Stage 1 #
    ###########
    # Estimates the performance of different classification algorithms on training data
    # X_train, y_train, X_test, y_test, _, _ = import_data_for_classification(training_spreadsheets_dir, data_dir, features='all')
    # find_best_estimator(X_train, y_train, results_dir)

    ###########
    # Stage 2 #
    ###########
    # Evaluates the final estimator on unseen data (i.e. estimator that best fits the problem)
    X_train, y_train, X_test, y_test, _, _, _ = import_data_for_classification(training_spreadsheets_dir, data_dir, features='all')
    evaluate_final_estimator_on_unseen_data(X_train, y_train, X_test, y_test, results_dir)

    # Using only statistic features
    # X_train, y_train, X_test, y_test, _, _, _ = import_data_for_classification(training_spreadsheets_dir, data_dir, features='statistic')
    # evaluate_final_estimator_on_unseen_data(X_train, y_train, X_test, y_test, results_dir)

    # Using only heuristic features
    # X_train, y_train, X_test, y_test, _, _, _ = import_data_for_classification(training_spreadsheets_dir, data_dir, features='heuristic')
    # evaluate_final_estimator_on_unseen_data(X_train, y_train, X_test, y_test, results_dir)

    # Evaluate usefulness of characteristics
    # X_train, y_train, _, _, _, _, selected_feature_names = import_data_for_classification(training_spreadsheets_dir, data_dir, features='all')
    # evaluate_usefulness_of_features(X_train, y_train, selected_feature_names, results_dir)
 
    ###########
    # Stage 3 #
    ###########
    # Train the final classification with all data and dump the model
    # train_final_estimator(X_train, y_train, X_test, y_test)
    # Predict samples using the final model for the survey evaluation
    # predict_survey_spreadsheets(spreadsheets_dir, survey_spreadsheets_dir, results_dir, 75)
    # predict_analysis_spreadsheets(os.path.join(spreadsheets_dir, 'for-prediction'),  os.path.join(spreadsheets_dir, 'for-analysis'), results_dir, 100)