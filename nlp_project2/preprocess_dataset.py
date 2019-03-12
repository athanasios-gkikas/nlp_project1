from collections import Counter
import os
from shutil import rmtree
import numpy
import random

def split_data(tweets, keys):
	new_tweets = {}

	for key in keys:
		new_tweets[key] = tweets[key]

	return new_tweets

def load_tsv(filepath):
	tweets = {}

	file = open(filepath, "r")
	for line in file:
		line = line.split("\t")
		#remove duplicates with the same label
		if line[0] in list(tweets.keys()) and line[1] != tweets[line[0]][0]:
			del tweets[line[0]]
		else:
			tweets[line[0]] = [line[1], line[2].replace("\n", "")]

	return tweets

def write_tsv(tweets, filename):
	output_file = open(filename, "w")

	for tweet in tweets:
		output_file.write(tweet + "\t" + tweets[tweet][0] + "\t" + tweets[tweet][1])
		output_file.write("\n")


def clean_tweets(tweets):
	cleaned_tweets = {}

	for tweet in tweets:
		if tweets[tweet][1] != "Not Available":
			cleaned_tweets[tweet] = tweets[tweet]
	return cleaned_tweets

def main():

	#load data
	train_path = "./twitter_download/train_set.tsv"
	dev_path = "./twitter_download/dev_set.tsv"
	test_path = "./twitter_download/test_set.tsv"

	train_tweets = load_tsv(train_path)
	dev_tweets = load_tsv(dev_path)
	test_tweets = load_tsv(test_path)

	print("Found", len(train_tweets),"in train set")
	print("Found", len(dev_tweets),"in dev set")
	print("Found", len(test_tweets),"in test set")

	#remove tweets that the test was not available
	cleaned_train_tweets = clean_tweets(train_tweets)
	cleaned_dev_tweets = clean_tweets(dev_tweets)
	cleaned_test_tweets = clean_tweets(test_tweets)

	print("Removed", len(train_tweets)-len(cleaned_train_tweets), "\"Not Available\" tweets from train set")
	print("Removed", len(dev_tweets)-len(cleaned_dev_tweets), "\"Not Available\" tweets from dev set")
	print("Removed", len(test_tweets)-len(cleaned_test_tweets), "\"Not Available\" tweets from test set")

	# print(list(cleaned_train_tweets.keys())[0])
	# print(cleaned_train_tweets[list(cleaned_train_tweets.keys())[0]][1])

	#split train data to train and val
	random.seed(42)
	keys = list(cleaned_train_tweets.keys())
	random.shuffle(keys)

	train_split = int(numpy.floor(len(cleaned_train_tweets) * 0.75))

	train_keys = keys[:train_split]
	val_keys = keys[train_split:]

	new_train_tweets = split_data(cleaned_train_tweets, train_keys)
	val_tweets = split_data(cleaned_train_tweets, val_keys)

	#create dataset folder
	try:
		rmtree('dataset/')
	except BaseException:
		pass  # directory doesn't yet exist, no need to clear it
	os.makedirs("dataset/")

	#write the resulted sets to tsv files
	write_tsv(new_train_tweets, "./dataset/train_set.tsv")
	write_tsv(val_tweets, "./dataset/val_set.tsv")
	write_tsv(cleaned_dev_tweets, "./dataset/dev_set.tsv")
	write_tsv(cleaned_test_tweets, "./dataset/test_set.tsv")

	print("Final dataset consists of:")
	print("Train set:", len(new_train_tweets))
	print("Val set:", len(val_tweets))
	print("Dev set:", len(cleaned_dev_tweets))
	print("Test set:", len(cleaned_test_tweets))



if __name__ == "__main__":
	main()