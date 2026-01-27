#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
import string

def text_preprocessing(X, techniques):
    """Applies text processing techniques to a dataframe column of strings (text).

    Before converting paragraphs into features, a good starting point may be to apply
    pre-processing techniques that will remove unwanted information. This method contains
    a series of submethods that represent common preprocessing techniques used in
    machine learning.
    
    X (Dataframe): Strings with raw text for classification.
    techniques (Dictionary): Keys representing techniques to be applied in the 
        text processing process.
    
    Returns:
        Dataframe: Column of strings updated with the values formated by the preprocessing
        techniques defined as input.
    """

    X = X.dropna()

    def lowercase(paragraph):
        # Transforms uppercase characters into lowercase
        return paragraph.lower()

    def remove_punctuations(paragraph):
        # Removes all the punctuations of the text, including: !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~
        return paragraph.translate(str.maketrans('', '', string.punctuation))

    def remove_stopwords(paragraph):
        # Removes all the stopwords of the paragraph, such as: "the, for, but, nor"
        stop_words = set(stopwords.words('english'))
        return " ".join([word for word in paragraph.split() if word not in stop_words])

    def stemming(paragraph):
        # Applies a stemmer technique for each word
        # Read about stemming at:
        # nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html
        stemmer = PorterStemmer()
        return " ".join([stemmer.stem(word) for word in paragraph.split()])

    def lemmatization(paragraph):
        # Applies a lemattizer for each word
        # Read about lemmatization at:
        # nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html
        lemmatizer = WordNetLemmatizer()
        return " ".join([lemmatizer.lemmatize(word) for word in paragraph.split()])

    if 'lowercase' in techniques:
        X = X.apply(lambda paragraph: lowercase(paragraph))

    if 'remove-punctuations' in techniques:        
        X = X.apply(lambda paragraph: remove_punctuations(paragraph))

    if 'remove-stopwords' in techniques:
        X = X.apply(lambda paragraph: remove_stopwords(paragraph))
    
    if 'stemming' in techniques:
        X = X.apply(lambda paragraph: stemming(paragraph))

    if 'lemmatization' in techniques:
        X = X.apply(lambda paragraph: lemmatization(paragraph))

    return X