# ==================================================
# File: classify.py
# Author: Trent Bultsma
# Date: 12/9/2022
# Description: Classifies tweet data as 
#   positive, negative, litigious, or uncertain
# ==================================================

import time
import pandas
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# read the data
tweet_data = pandas.read_csv('dataset_sanitized.csv', sep=',')

# filter out just the english data (no spanish or french just to make it consistent)
english_tweet_data = tweet_data.loc[np.where(tweet_data['Language'] == 'en')]

# split the data into training and testing data
training_text, testing_text, training_label, testing_label = train_test_split(
    english_tweet_data['Text'], english_tweet_data['Label'], test_size=0.2, random_state=0)

classifiers = [
    ('Support Vector Machine', LinearSVC()),
    ('Naive Bayes', MultinomialNB()),
    ('Decision Tree', DecisionTreeClassifier(max_depth=3)),
]

# for keeping track of the most accurate classifier
best_classifier = None
best_classifier_accuracy = 0

# go through the classifiers
for classifier_name, classifier in classifiers:

    start_time = time.time()

    # setup the pipeline for processing data
    text_classifier = Pipeline([
        ('vect', CountVectorizer()), # creates a dictionary of words in the tweet
        ('tfidf', TfidfTransformer()), # normalize word occurance count (by tweet length)
        ('clf', classifier), # the actual classifier
    ])

    # train the classifier
    text_classifier.fit(training_text, training_label)

    # test the classifier
    predicted = text_classifier.predict(testing_text)
    accuarcy = np.mean(predicted == testing_label)

    # display the stats
    print('-- ' + classifier_name + ' --')
    print('Accuracy: {:.2f}%'.format(accuarcy * 100))
    print('Calculation Time: {:.2f} seconds'.format(time.time() - start_time))
    print()

    # update the best classifier
    if accuarcy > best_classifier_accuracy:
        best_classifier_accuracy = accuarcy
        best_classifier = text_classifier

# have manual input at the end for the best one
while True:
    sentence = input('Input a sentence to test: ')
    print('Classified as ' + best_classifier.predict([sentence])[0])