# ==================================================
# File: classify.py
# Author: Trent Bultsma
# Date: 12/9/2022
# Description: Classifies tweet data as 
#   positive, negative, litigious, or uncertain
# ==================================================

import pandas
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# read the data
try:
    tweet_data = pandas.read_csv('dataset.csv', sep=',')
except FileNotFoundError as e:
    print(e)
    exit()

# filter out just the english data (no spanish or french just to make it consistent)
english_tweet_data = tweet_data.loc[np.where(tweet_data['Language'] == 'en')]

# TODO: perhaps take out hashtags, tagging people, and urls here

# split the data into training and testing data
training_text, testing_text, training_label, testing_label = train_test_split(
    english_tweet_data["Text"], english_tweet_data["Label"], test_size=0.2, random_state=0)

# setup the pipeline for processing data
text_classifier = Pipeline([
    ('vect', CountVectorizer()), # creates a dictionary of words in the tweet
    ('tfidf', TfidfTransformer()), # normalize word occurance count (by tweet length)
    ('clf', MultinomialNB()), # the actual classifier
])

# train the classifier
text_classifier.fit(training_text, training_label)

# test the classifier
predicted = text_classifier.predict(testing_text)
print(np.mean(predicted == testing_label))

while True:
    sentence = input("Input a sentence to test: ")
    print("Classified as " + text_classifier.predict([sentence])[0])