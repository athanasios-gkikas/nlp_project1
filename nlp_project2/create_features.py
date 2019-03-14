import re
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords
from sklearn import preprocessing
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pprint
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import pandas as pd

def load_tsv(filepath):
	tweets = {}

	file = open(filepath, "r")
	for line in file:
		line = line.split("\t")
		tweets[line[0]] = [line[1], line[2].replace("\n", "")]

	return tweets

def write_tsv(tweets, filename):
	output_file = open(filename, "w")

	for tweet in tweets:
		output_file.write(tweet + "\t" + tweets[tweet][0] + "\t" + " ".join(tweets[tweet][1]))
		output_file.write("\n")

def process_tweets(tweets):
	tweet_stopwords = set(stopwords.words('english') + list(punctuation) + ["..."] + ['AT_USER','URL'])
	new_tweets = {}

	for tweet in tweets:
		text = tweets[tweet][1].lower() # convert text to lower-case
		text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text) # remove URLs
		text = re.sub('@[^\s]+', 'AT_USER', text) # remove usernames
		text = re.sub(r'#([^\s]+)', r'\1', text) # remove the # in #hashtag
		text = re.sub('([.]+)', '.', text) #remove multiple dots
		text = re.sub(r"(?:['\-_]+[a-z])", ' ', text) #remove single characters with ' or -
		text = word_tokenize(text)
		text = [word for word in text if word not in tweet_stopwords]

		new_tweets[tweet] = [tweets[tweet][0], " ".join(text)]

	return new_tweets


def main():

	train_tweets = load_tsv("./dataset/train_set.tsv")
	val_tweets = load_tsv("./dataset/val_set.tsv")
	dev_tweets = load_tsv("./dataset/dev_set.tsv")
	test_tweets = load_tsv("./dataset/test_set.tsv")

	pr_train_tweets = process_tweets(train_tweets)
	pr_val_tweets = process_tweets(val_tweets)
	pr_dev_tweets = process_tweets(dev_tweets)
	pr_test_tweets = process_tweets(test_tweets)

	# #create processed dataset folder
	# try:
	# 	rmtree("processed_dataset/")
	# except BaseException:
	# 	pass  # directory doesn't yet exist, no need to clear it
	# os.makedirs("processed_dataset/")

	# #save processed tweets
	# write_tsv(pr_train_tweets, "./processed_dataset/train_set.tsv")
	# write_tsv(pr_val_tweets, "./processed_dataset/val_set.tsv")
	# write_tsv(pr_dev_tweets, "./processed_dataset/dev_set.tsv")
	# write_tsv(pr_test_tweets, "./processed_dataset/test_set.tsv")

	#get text of train and dev sets
	x_train = [tweet[1] for tweet in list(pr_train_tweets.values())]
	x_dev = [tweet[1] for tweet in list(pr_dev_tweets.values())]

	#get train and dev labels and transform them to one hot encoding
	y_train = [tweet[0] for tweet in list(pr_train_tweets.values())]
	le = preprocessing.LabelEncoder()
	le.fit(y_train)
	y_train = le.transform(y_train)
	y_dev = le.transform([tweet[0] for tweet in list(pr_dev_tweets.values())])

	vectorizer = TfidfVectorizer(ngram_range = (1,2), max_features = 5000, sublinear_tf = True)

	x_train_tfidf = vectorizer.fit_transform(x_train)
	x_dev_tfidf = vectorizer.transform(x_dev)

	#pprint.pprint(vectorizer.get_feature_names())

	# Logistic Regression classifier
	clf = LogisticRegression(solver="liblinear")
	clf.fit(x_train_tfidf, y_train)
	predictions = clf.predict(x_train_tfidf)
	score = f1_score(y_train, predictions, average = None)
	print("train f1-score:", score)

	predictions_dev = clf.predict(x_dev_tfidf)
	score = f1_score(y_dev, predictions_dev, average = None)
	print("dev f1-score:", score)
	print()
	# print("dev data confusion matrix:")

	# y_true = pd.Series(y_dev, name = "True")
	# y_pred = pd.Series(predictions_dev, name = "Predicted")

	# pd.crosstab(y_true, y_pred)


if __name__ == "__main__":
	main()