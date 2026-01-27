#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import os
import numpy as np
np.set_printoptions(threshold=np.inf)
from imblearn.pipeline import Pipeline

# Cross-validation
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV

# Oversampling techniques
from imblearn.over_sampling import SMOTE

# Training strategies
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier

# Classification Algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

def evaluate_estimators_performance(classifiers, strategies, oversample, 
                                    X_train, y_train, results_dir):
    """For a set of pre-defined estimators, evaluate
    their performance on training instances.

    Args:
        classifiers (List of strings): The list of classifiers
        which performance should be estimated. 
        strategies (List of strings): The list of multiclass
        estrategies that should be used during estimation.
        oversample (Boolean): The possibility (or not) of
        using an oversampling technique on training data.
        X_train (Dataframe): Training features
        y_train (Series): Training labels
        results_dir (String): Folder where results
        should be saved.
    """

    # Dictionary of classification algorithms available
    classifiers_available = {
        'rf': RandomForestClassifier(n_jobs=-1),
        'svc': LinearSVC(),
        'mnb': MultinomialNB(),
        'knn': KNeighborsClassifier(),
        'lr': LogisticRegression(),
        'dmr': DummyClassifier(strategy='uniform', random_state=42),
        'dmf': DummyClassifier(strategy='most_frequent')
    }

    # Dictionary of hyperpameters tested for each
    # classification algorithm above.
    hyperparameters_available = {
        'rf': {'clf__estimator__n_estimators': [75, 100, 125]},
        'svc': {'clf__estimator__tol': [1e-3, 1e-4, 1e-5],
                'clf__estimator__C': [0.5, 1, 1.5],
                'clf__estimator__max_iter': [500, 1000, 1500]},
        'mnb': {'clf__estimator__alpha': [0.5, 1, 1.5],
                'clf__estimator__fit_prior': [True, False]},
        'knn': {'clf__estimator__n_neighbors': [3, 5, 7]},
        'lr': {'clf__estimator__tol': [1e-3, 1e-4, 1e-5],
               'clf__estimator__C': [0.5, 1, 1.5]},
        'dmr': {}, # This is a dummy classifier, no hyperparameters are used.
        'dmf': {}  # This is a dummy classifier, no hyperparameters are used.
    }

    # Dictionary of multiclass strategies used
    # during training.
    strategies_available = {
        'ovr': 'one_vs_rest',
        'ovo': 'one_vs_one'
    }

    # Identifies which resources the user decided to use.
    # Resources here mean: classification algorithms, different
    # hyperparameters and training strategies (including oversampling).
    selected_classifiers = [classifiers_available[classifier]
                            for classifier in classifiers]

    selected_hyperparameters = [hyperparameters_available[classifier]
                                for classifier in classifiers]

    selected_strategies = [strategies_available[strategy]
                            for strategy in strategies]

    # For all the selected resources, estimate the performance and
    # export the results.
    for classifier, hyperparameters in zip(selected_classifiers, selected_hyperparameters):
        classifier_name = type(classifier).__name__
        print("Evaluating estimator: " + classifier_name)

        for strategy in selected_strategies:
            print("Multiclass strategy: " + strategy)

            for oversample_condition in oversample:
                oversample_bool = str(oversample_condition).lower()
                print("Oversample: " + oversample_bool)

                estimator_args = {
                    'classifier': classifier,
                    'hyperparameters': hyperparameters,
                    'strategy': strategy,
                    'oversample': oversample_condition,
                    'X_train': X_train,
                    'y_train': y_train
                }

                internal_evaluation, f1_weighted_scores_means = nested_cross_validation(**estimator_args)

                export_estimator_results(estimator_args, internal_evaluation, 
                                         f1_weighted_scores_means, results_dir)


def nested_cross_validation(classifier, hyperparameters, strategy, 
                            oversample, X_train, y_train):
    """Computes a multiclass nested ten-fold cross-validation.

    To better evaluate how each algorithm performs in our data,
    a nested stratified k-folding cross-validation approach was applied.

    Args:
        classifier (Class): An instance of a scikit-learn classifier.
        hyperparameters (Dictionary): Hyperparameters to be used in GridSearch.
        strategy (String): Defines which strategies will be used during training.
        oversample (Boolean): Defines if smote oversampling should be applied.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        F1 weighted scores from cross-validation.
    """

    # Selected resources are used
    # in a scikit-learn pipeline
    pipeline_args = []

    # IF oversample is selected
    if oversample:
        oversample = ('smt', SMOTE())
        pipeline_args.append(oversample)

    # If one of the following multiclass
    # strategies is elected.
    if strategy == 'one_vs_rest':
        strategy_arg = ('clf', OneVsRestClassifier(classifier))
        pipeline_args.append(strategy_arg)
    elif strategy == 'one_vs_one':
        strategy_arg = ('clf', OneVsOneClassifier(classifier))
        pipeline_args.append(strategy_arg)

    pipeline = Pipeline(pipeline_args)

    # The internal cross-validation is used for hyperparameter optimization.
    print('Running hyperparameter optimization.')
    internal_cv = StratifiedKFold(n_splits=10)
    internal_evaluation = GridSearchCV(pipeline, hyperparameters, cv=internal_cv)
    internal_evaluation.fit(X_train, y_train)

    # The external cross-validation is used for model selection.
    external_cv = StratifiedKFold(n_splits=10)
    f1_weighted_scores_means = []

    # We perform the external cross-validation `num` times to 
    # evaluate difference of the average scores of each model
    # in OneVsRest strategy. The result will be always the same
    # for OneVsOne.
    # See thread: https://stats.stackexchange.com/questions/91091/one-vs-all-and-one-vs-one-in-svm

    num_times = 1

    if strategy == 'one_vs_rest':
        num_times = 10

    for times in range(0, num_times):
        print('Running model selection cross-validation (' + str(times + 1) + ' out of ' + str(num_times) + ')')
        # Notice that the variable `internal_evaluation` represents 
        # the classification model with the best hyperparameters.
        f1_weighted_scores = cross_val_score(internal_evaluation, X_train, y_train, scoring='f1_weighted', cv=external_cv)
        f1_weighted_scores_means.append(f1_weighted_scores.mean())

    return internal_evaluation, f1_weighted_scores_means

def export_estimator_results(estimator_args, internal_evaluation, f1_weighted_scores_means, results_dir):
    """Exports a estimators performance to a text file.

    Args:
        estimator_args (Dictionary): Arguments used to train the classifier (e.g. training strategy)
        internal_evaluation (Class): GridSearch results
        f1_weighted_scores_means (List): F1 weighted scores (mean)
        results_dir (String): Folder whre results should be saved.
    """

    filename = type(estimator_args['classifier']).__name__ + '_strategy_' + \
               estimator_args['strategy'] + '_oversample_' + str(estimator_args['oversample']).lower()

    filepath = os.path.join(results_dir, filename)
    
    # If file already exists, don't replace it.
    if os.path.exists(filepath):
        num_attempts = 1

        while True:
            new_filepath = filepath + ' (' + num_attempts + ')'

            if not os.path.exists(new_filepath):
                filepath = new_filepath
                break
            else:
                num_attempts = num_attempts + 1

    with open(filepath, 'w') as results:
        # Estimator settings
        results.write('Estimator: ' + type(estimator_args['classifier']).__name__ + '\n')
        results.write('Strategy: ' + estimator_args['strategy'] + '\n')
        results.write('Oversample: ' + str(estimator_args['oversample']) + '\n\n')
        # Hyperparameter optimization report
        results.write('Internal Cross Validation (GridSearch)\n')
        results.write('Best parameters: ' + str(internal_evaluation.best_params_) + '\n')
        results.write('Best estimator: ' + str(internal_evaluation.best_estimator_) + '\n')
        # Model selection report 
        results.write('External Cross Validation (cross_val_score)\n')
        results.write('F1 weighted scores (averages): \n')

        for mean in f1_weighted_scores_means:
            results.write(str(mean) + '\n')
