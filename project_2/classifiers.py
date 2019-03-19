import re
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords
from sklearn import preprocessing
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
import pandas as pd

def load_tsv(filepath):
	tweets = {}

	with open(filepath, "r") as file:
		for line in file:
			line = line.split("\t")
			tweets[line[0]] = [line[1], line[2].replace("\n", "")]

	return tweets

def process_tweets(tweets):
	tweet_stopwords = set(stopwords.words('english') + list(punctuation) + ["..."] + ['AT_USER','URL'])
	new_tweets = {}

	for tweet in tweets:
		text = tweets[tweet][1].lower() # convert text to lower-case
		text = find_emoticons_in_tweets(text) # convert emoticons to text
		text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text) # remove URLs
		text = re.sub('@[^\s]+', 'AT_USER', text) # remove usernames
		text = re.sub(r'#([^\s]+)', r'\1', text) # remove the # in #hashtag
		text = re.sub('([.]+)', '.', text) #remove multiple dots
		text = re.sub(r"(?:['\-_]+[a-z])", ' ', text) #remove single characters with ' or -
		text = word_tokenize(text)
		text = [word for word in text if word not in tweet_stopwords]

		new_tweets[tweet] = [tweets[tweet][0], " ".join(text)]

	return new_tweets

def find_emoticons_in_tweets(tweet):
	new_tweet = tweet
	# All caps? Does it matter?
	repl = {' :)': ' HAPPY_EMOTICON', ' =)': ' HAPPY_EMOTICON', ' :d': ' VERY_HAPPY_EMOTICON', ' :(': ' SAD_EMOTICON', ' :/': ' MIXED_EMOTICON', ' :p': ' TONGUE_EMOTICON', ' ;)': ' WINK_EMOTICON'}
	for a, b in repl.items():
		new_tweet = new_tweet.replace(a, b)

	return new_tweet

def main():

	train_tweets = load_tsv("./dataset/train_set.tsv")
	val_tweets = load_tsv("./dataset/val_set.tsv")
	dev_tweets = load_tsv("./dataset/dev_set.tsv")
	test_tweets = load_tsv("./dataset/test_set.tsv")

	pr_train_tweets = process_tweets(train_tweets)
	pr_val_tweets = process_tweets(val_tweets)
	pr_dev_tweets = process_tweets(dev_tweets)
	pr_test_tweets = process_tweets(test_tweets)

	#get text of train and dev sets
	x_train = [tweet[1] for tweet in list(pr_train_tweets.values())]
	x_dev = [tweet[1] for tweet in list(pr_dev_tweets.values())]

	#get train and dev labels and transform them to numerical
	y_train = [tweet[0] for tweet in list(pr_train_tweets.values())]
	le = preprocessing.LabelEncoder()
	le.fit(y_train)
	y_train = le.transform(y_train)
	y_dev = le.transform([tweet[0] for tweet in list(pr_dev_tweets.values())])

	#check parameters
	#tune ngram_range, max_features
	vectorizer = TfidfVectorizer(ngram_range = (1,2), max_features = 5000, sublinear_tf = True)

	x_train_tfidf = vectorizer.fit_transform(x_train)
	x_dev_tfidf = vectorizer.transform(x_dev)

	#add evaluation for all measures (like shown in the lab tutorial)
	#add curves

	#------------------------Most frequent classifier-----------------------------
	baseline = DummyClassifier(strategy = "most_frequent")
	baseline.fit(x_train_tfidf, y_train)
	#predict for train
	predictions_train = baseline.predict(x_train_tfidf)
	score = f1_score(y_train, predictions_train, average = "macro")
	print("train f1-score:", score)

	#predict for dev
	predictions_dev = baseline.predict(x_dev_tfidf)
	score = f1_score(y_dev, predictions_dev, average = "macro")
	print("dev f1-score:", score)

	#-------------------Logistic Regression classifier------------------------
	clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight = "balanced")
	clf.fit(x_train_tfidf, y_train)
	#predict for train
	predictions = clf.predict(x_train_tfidf)
	score = f1_score(y_train, predictions, average = "macro")
	print("train f1-score:", score)

	#predict for dev
	predictions_dev = clf.predict(x_dev_tfidf)
	score = f1_score(y_dev, predictions_dev, average = "macro")
	print("dev f1-score:", score)



if __name__ == "__main__":
	main()