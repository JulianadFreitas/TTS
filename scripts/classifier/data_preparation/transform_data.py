#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import os
import pandas

def transform_spreadsheets_in_dataframe(spreadsheets_dir, text_column, 
                                        classes_columns, label_column):
    """Transforms annotated spreadsheets in a dataframe structure.

    Args:
        spreadsheets_dir (String): Represents the path to the directory where the
            spreadsheets are located.

    Returns:
        Dataframe: Contains the parsed spreadsheets in a dataframe structure.
    """
    dataframe = pandas.DataFrame()

    for filename in os.listdir(spreadsheets_dir):
        filepath = os.path.join(spreadsheets_dir, filename)

        if os.path.isfile(filepath):
            if filename.endswith('.xlsx'):
                rows = parse_spreadsheet_file(filepath, text_column,
                                              classes_columns, label_column)
                dataframe = pandas.concat([dataframe, rows])

    return dataframe

def parse_spreadsheet_file(filepath, text_column, classes_columns, label_column):
    """Extracts the annotated data from one spreadsheet file.

    Args:
        filepath (String): Represents the path to a spreadsheet.
        classes (List of strings): Contains the columns that should be extracted
            from the spreadsheet as classes of the classifier.

    Returns:
       Dataframe: Contains the data from the spreadsheet.
    """
    spreadsheet = pandas.ExcelFile(filepath, engine='openpyxl')
    dataframe = pandas.DataFrame()

    for worksheet_name in spreadsheet.sheet_names:
        worksheet = spreadsheet.parse(worksheet_name)

        # Identify the text column with text_column label
        worksheet.rename(columns={worksheet.columns[0]: text_column}, inplace = True)
        worksheet[text_column].astype(str)

        is_predict = False

        if label_column == 'Prediction':
            owner, name = filepath.split('\\')[-1].split('@')
            is_predict = 'https://github.com/' + owner + '/' + name.replace('.xlsx', '')

        # Replace NaNs with 0s and non NaNs with 1s
        for column in classes_columns:
            if column in worksheet:
                worksheet[column] = worksheet[column].notnull().astype(int)
                
        worksheet[label_column] = worksheet.apply(lambda row: 
                                                  define_label(row, classes_columns, is_predict),
                                                  axis=1)
        worksheet['Spreadsheet'] = os.path.basename(filepath)
        worksheet['Worksheet'] = worksheet_name
        worksheet['Row Index'] = worksheet.index
        dataframe = pandas.concat([worksheet, dataframe])

    return dataframe

def define_label(row, classes_columns, is_predict):
    """Identifies which label should be assigned to a row in a spreadsheet

    This study solves a multiclass classification problem. For this reason, a 
    column named as `label` is created for each row in a spreadsheet file. 
    This method identifies which one of the columns representing classes of the
    problem is the one that should be assigned as a label for the respective
    respective row. 

    Args:
        row: A pandas dataframe row. 
        classes: A list of strings containing the columns that should be extracted
            from the spreadsheet as classes of the classifier.
    
    Returns:
        A string representing the label for the respective row.
    """
    if is_predict is not False:
        return is_predict

    label = 'No categories identified.'

    for _class in classes_columns:
        if _class in row:
            # If row was annotated with
            # the respective class
            if row[_class] == 1:
                label = _class

    return label