# Text Sentiment Classifier

## Data Mining Task: 
The task is to classify the sentiment of a tweet, being either positive, negative, litigious, or uncertain. Using tweets as a baseline, this sentiment classification could be potentially extended to other text based content. The most useful application in industry for this would be something like monitoring the sentiment of a product or service, serving as a thermometer for customer satisfaction for a company.

## Dataset: 
The dataset accessible on [Kaggle](https://www.kaggle.com/datasets/tariqsays/sentiment-dataset-with-1-million-tweets) contains the text of roughly one million tweets tagged with the language and sentiment of the tweet. 

## Methodology: 
I will use a naïve bayes classification solution using the scikit-learn library to split the tweets into the 4 categories of sentiment. I will develop the solution using a smaller test dataset of a few thousand tweets until I get the basics worked out and then train it on 80 percent of the tweets (being 800 thousand) then use the remaining 20 percent as test data to measure the accuracy of my solution. I am planning to experiment using the removal of stop words like “and” and “the” to improve efficiency but may not end up doing that as some of them can change the meaning of the sentence which is important for sentiment analysis. Scikit has a library of these words to use.

## Final product:
The project will be successful if it has a high rate of accuracy in predicting the sentiment of tweets in the test data set (say 90% or something. The number is arbitrary but as long as it is high, we are good). I will learn more about the naïve bayes algorithm through this project as well as improving my ability to implement machine learning solutions as I haven’t had any experience on that at all from before this course.

## Instructions:
Clone the repository.

Install the required libraries:
```
pip install pandas
pip install scikit-learn
```
Run `setup_data.py` then `classify.py`
