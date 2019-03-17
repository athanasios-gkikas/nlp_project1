from collections import Counter
import os
from shutil import rmtree
import numpy
import random
from collections import Counter

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

	print("After removing duplicates and tweets that are the same but have different labels:")
	print(len(train_tweets),"remained in train set")
	print(len(dev_tweets),"remained in dev set")
	print(len(test_tweets),"remained in devtest set")

	#remove tweets that the test was not available
	cleaned_train_tweets = clean_tweets(train_tweets)
	cleaned_dev_tweets = clean_tweets(dev_tweets)
	cleaned_test_tweets = clean_tweets(test_tweets)

	print("Removed", len(train_tweets)-len(cleaned_train_tweets), "\"Not Available\" tweets from train set")
	print("Removed", len(dev_tweets)-len(cleaned_dev_tweets), "\"Not Available\" tweets from dev set")
	print("Removed", len(test_tweets)-len(cleaned_test_tweets), "\"Not Available\" tweets from test set")


	print("Merging existing sets...")

	print("There are", len(cleaned_train_tweets) + len(cleaned_dev_tweets) + len(cleaned_test_tweets), "in total")

	#merge and remove duplicates with different label
	#improve later
	total_tweets = cleaned_train_tweets
	ids = list(cleaned_train_tweets.keys())

	for dev_tweet in cleaned_dev_tweets:
		if dev_tweet in ids and cleaned_dev_tweets[dev_tweet][0] != total_tweets[dev_tweet][0]:
			del total_tweets[dev_tweet]
		else:
			total_tweets[dev_tweet] = cleaned_dev_tweets[dev_tweet]

	ids = list(total_tweets.keys())

	for test_tweet in cleaned_test_tweets:
		if test_tweet in ids and cleaned_test_tweets[test_tweet][0] != total_tweets[test_tweet][0]:
			del total_tweets[test_tweet]
		else:
			total_tweets[test_tweet] = cleaned_test_tweets[test_tweet]

	print("Merged set has", len(total_tweets), "tweets")

	labels = [total_tweets[tweet][0] for tweet in total_tweets]

	count_labels = Counter(labels)

	print(count_labels)

	#split train data to train, val, dev and test sets
	#split randomly, change later
	random.seed(42)
	keys = list(total_tweets.keys())
	random.shuffle(keys)

	train_split = int(numpy.floor(len(total_tweets) * 0.7))
	other_splits = int(numpy.floor(len(total_tweets) * 0.1))

	train_keys = keys[:train_split]
	val_keys = keys[train_split:train_split+other_splits]
	dev_keys = keys[train_split+other_splits:train_split+other_splits+other_splits]
	test_keys = keys[train_split+other_splits+other_splits:]

	train_tweets = split_data(total_tweets, train_keys)
	val_tweets = split_data(total_tweets, val_keys)
	dev_tweets = split_data(total_tweets, dev_keys)
	test_tweets = split_data(total_tweets, test_keys)

	#create dataset folder
	try:
		rmtree('dataset/')
	except BaseException:
		pass  # directory doesn't yet exist, no need to clear it
	os.makedirs("dataset/")

	#write the resulted sets to tsv files
	write_tsv(train_tweets, "./dataset/train_set.tsv")
	write_tsv(val_tweets, "./dataset/val_set.tsv")
	write_tsv(dev_tweets, "./dataset/dev_set.tsv")
	write_tsv(test_tweets, "./dataset/test_set.tsv")

	print("Final dataset consists of:")
	print("Train set:", len(train_tweets))
	print("Val set:", len(val_tweets))
	print("Dev set:", len(dev_tweets))
	print("Test set:", len(test_tweets))



if __name__ == "__main__":
	main()