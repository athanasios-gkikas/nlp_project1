import re
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from sklearn import preprocessing
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier


def load_tsv(filepath):
	tweets = {}

	with open(filepath, "r") as file:
		for line in file:
			line = line.split("\t")
			tweets[line[0]] = [line[1], line[2].replace("\n", "")]

	return tweets


def process_tweets(tweets):
	tweet_stopwords = set(stopwords.words('english') + list(punctuation) + ["..."] + ['AT_USER', 'URL'])
	new_tweets = {}

	for tweet in tweets:
		text = tweets[tweet][1].lower()  # convert text to lower-case
		text = find_emoticons_in_tweets(text)  # convert emoticons to text
		text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)  # remove URLs
		text = re.sub('@[^\s]+', 'AT_USER', text)  # remove usernames
		text = re.sub(r'#([^\s]+)', r'\1', text)  # remove the # in #hashtag
		text = re.sub('([.]+)', '.', text)  # remove multiple dots
		text = re.sub(r"(?:['\-_]+[a-z])", ' ', text)  # remove single characters with ' or -
		text = word_tokenize(text)
		text = [word for word in text if word not in tweet_stopwords]

		new_tweets[tweet] = [tweets[tweet][0], " ".join(text)]

	return new_tweets


def find_emoticons_in_tweets(tweet):
	new_tweet = tweet
	# All caps? Does it matter?
	repl = {' :)': ' HAPPY_EMOTICON', ' =)': ' HAPPY_EMOTICON', ' :d': ' VERY_HAPPY_EMOTICON', ' :(': ' SAD_EMOTICON',
	        ' :/': ' MIXED_EMOTICON', ' :p': ' TONGUE_EMOTICON', ' ;)': ' WINK_EMOTICON'}
	for a, b in repl.items():
		new_tweet = new_tweet.replace(a, b)

	return new_tweet


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10), scoring='f1_macro'):
	plt.figure()
	plt.title(title)
	if ylim is not None:
		plt.ylim(*ylim)
	plt.xlabel("Training examples")
	plt.ylabel("Score")
	train_sizes, train_scores, test_scores = learning_curve(
		estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
		scoring=scoring)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	plt.grid()

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
	                 train_scores_mean + train_scores_std, alpha=0.1,
	                 color="r")
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
	                 test_scores_mean + test_scores_std, alpha=0.1, color="b")
	plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
	         label="Training score")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="b",
	         label="Test score")

	plt.legend(loc="best")
	return plt


def plot_precision_recall_curves(estimator, name, x_train, y_1, x_test, y_2):
	y_train = label_binarize(y_1, classes=[0, 1, 2])
	y_test = label_binarize(y_2, classes=[0, 1, 2])

	labels = ["negative", "neutral", "positive"]

	classifier = OneVsRestClassifier(estimator)
	y_score = classifier.fit(x_train, y_train).decision_function(x_test)

	# Compute Precision-Recall and plot curve
	precision = dict()
	recall = dict()
	average_precision = dict()
	n_classes = 3
	for i in range(n_classes):
		precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
		                                                    y_score[:, i])
		average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
	# Compute micro-average ROC curve and ROC area
	precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
	                                                                y_score.ravel())
	average_precision["micro"] = average_precision_score(y_test, y_score,
	                                                     average="micro")

	# Plot Precision-Recall curve

	plt.clf()
	plt.plot(recall["micro"], precision["micro"], color='navy',
	         label='micro-average Precision-recall curve (area = {0:0.2f})'
	               ''.format(average_precision["micro"]))
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title('Precision-Recall {0}: AUC={1:0.2f}'.format(name, average_precision['micro']))
	plt.legend(loc="lower left")
	plt.show()

	# Plot Precision-Recall curve for each class
	plt.clf()
	plt.plot(recall["micro"], precision["micro"], color='navy',
	         label='micro-average Precision-recall curve (area = {0:0.2f})'
	               ''.format(average_precision["micro"]))
	for i, color in zip(range(3), ['red', 'pink', 'blue']):
		plt.plot(recall[i], precision[i], color=color,
		         label='Precision-recall curve of class {0} (area = {1:0.2f})'
		               ''.format(labels[i], average_precision[i]))

	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Extension of Precision-Recall curve to multi-class ({0})'.format(name))
	plt.legend(loc="lower right")
	plt.show()
	return


def main():
	train_tweets = load_tsv("./dataset/train_set.tsv")
	test_tweets = load_tsv("./dataset/test_set.tsv")

	pr_train_tweets = process_tweets(train_tweets)
	pr_test_tweets = process_tweets(test_tweets)

	# Get text of train and dev sets
	x_train = [tweet[1] for tweet in list(pr_train_tweets.values())]
	x_test = [tweet[1] for tweet in list(pr_test_tweets.values())]

	# Get train and dev labels and transform them to numerical
	y_train = [tweet[0] for tweet in list(pr_train_tweets.values())]
	le = preprocessing.LabelEncoder()
	le.fit(y_train)
	y_train = le.transform(y_train)
	y_test = le.transform([tweet[0] for tweet in list(pr_test_tweets.values())])

	# check parameters
	# tune ngram_range, max_features
	vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)

	x_train_tfidf = vectorizer.fit_transform(x_train)
	x_test_tfidf = vectorizer.transform(x_test)

	# add evaluation for all measures (like shown in the lab tutorial)
	# add curves

	# ------------------------Most frequent classifier-----------------------------
	baseline = DummyClassifier(strategy="most_frequent")
	baseline.fit(x_train_tfidf, y_train)
	# Predict for train
	predictions_train = baseline.predict(x_train_tfidf)
	score = f1_score(y_train, predictions_train, average="macro")
	print("train f1-score:", score)

	# Predict for dev
	predictions_test = baseline.predict(x_test_tfidf)
	score = f1_score(y_test, predictions_test, average="macro")
	print("dev f1-score:", score)

	# -------------------Logistic Regression classifier------------------------
	clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight="balanced")
	clf.fit(x_train_tfidf, y_train)
	# Predict for train
	predictions = clf.predict(x_train_tfidf)
	score = f1_score(y_train, predictions, average="macro")
	print("train f1-score:", score)

	# Predict for dev
	predictions_test = clf.predict(x_test_tfidf)
	score = f1_score(y_test, predictions_test, average="macro")
	print("dev f1-score:", score)

	plot_learning_curve(baseline, "Learning Curves for DummyClassifier", x_train_tfidf, y_train, (0.1, 1.01), cv=None,
	                    n_jobs=-1).show()
	plot_learning_curve(clf, "Learning Curves for Logistic Regression", x_train_tfidf, y_train, (0.1, 1.01), cv=None,
	                    n_jobs=-1).show()
	plot_precision_recall_curves(clf, "Logistic Regression", x_train_tfidf, y_train, x_test_tfidf, y_test)


if __name__ == "__main__":
	main()
