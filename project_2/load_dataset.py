import os
import numpy
import random
import re

from shutil import rmtree
from collections import Counter
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords

def split_data(tweets, keys):
	new_tweets = {}

	for key in keys:
		new_tweets[key] = tweets[key]

	return new_tweets

def load_tsv(filepath):
	tweets = {}
	different_duplicates = []

	with open(filepath, "r") as file:
		for line in file:
			line = line.split("\t")
			#if this tweet already exists but with a different label don't add it
			if line[0] in list(tweets.keys()) and line[1] != tweets[line[0]][0]:
				different_duplicates.append(line[0])
			else:
				tweets[line[0]] = [line[1], line[2].replace("\n", "")]

	#remove tweets that have duplicates with different label
	for duplicate in different_duplicates:
		del tweets[duplicate]

	return tweets

def write_tsv(tweets, filename):

	with open(filename, "w") as output_file:
		for tweet in tweets:
			output_file.write(tweet + "\t" + tweets[tweet][0] + "\t" + tweets[tweet][1])
			output_file.write("\n")


def clean_tweets(tweets):
	cleaned_tweets = {}

	for tweet in tweets:
		if tweets[tweet][1] != "Not Available":
			cleaned_tweets[tweet] = tweets[tweet]
	return cleaned_tweets

def merge(first_set, second_set):
	total = first_set
	different_duplicates = []

	for tweet in second_set:
		#if this tweet already exists but with a different label don't add it
		if tweet in list(total.keys()) and second_set[tweet][0] != total[tweet][0]:
			different_duplicates.append(tweet)
		else:
			total[tweet] = second_set[tweet]

	#remove tweets that have duplicates with different label
	for duplicate in different_duplicates:
		del total[duplicate]

	return total

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

def main():

    #load data
    train_path = "twitter_download/train_set.tsv"
    dev_path = "twitter_download/dev_set.tsv"
    test_path = "twitter_download/test_set.tsv"

    train_tweets = load_tsv(train_path)
    dev_tweets = load_tsv(dev_path)
    test_tweets = load_tsv(test_path)

    print("After removing duplicates and tweets that are the same but have different labels:")
    print(len(train_tweets),"remained in train set")
    print(len(dev_tweets),"remained in dev set")
    print(len(test_tweets),"remained in devtest set")

    #remove tweets that the text was not available
    cleaned_train_tweets = clean_tweets(train_tweets)
    cleaned_dev_tweets = clean_tweets(dev_tweets)
    cleaned_test_tweets = clean_tweets(test_tweets)

    print("Removed", len(train_tweets)-len(cleaned_train_tweets), "\"Not Available\" tweets from train set")
    print("Removed", len(dev_tweets)-len(cleaned_dev_tweets), "\"Not Available\" tweets from dev set")
    print("Removed", len(test_tweets)-len(cleaned_test_tweets), "\"Not Available\" tweets from test set")


    print("Merging existing sets...")

    print("There are", len(cleaned_train_tweets) + len(cleaned_dev_tweets) + len(cleaned_test_tweets), "in total")

    #merge the sets removing duplicates with different label
    train_dev = merge(cleaned_train_tweets, cleaned_dev_tweets)
    total_tweets = merge(train_dev, cleaned_test_tweets)

    print("Merged set has", len(total_tweets), "tweets")

    labels = [total_tweets[tweet][0] for tweet in total_tweets]

    count_labels = Counter(labels)

    print(count_labels)

    #split randomly to train, val, dev and test sets
    random.seed(42)
    keys = list(total_tweets.keys())
    random.shuffle(keys)

    train_split = int(numpy.floor(len(total_tweets) * 0.8))

    train_keys = keys[:train_split]
    test_keys = keys[train_split:]

    train_tweets = split_data(total_tweets, train_keys)
    test_tweets = split_data(total_tweets, test_keys)

    #create dataset folder
    try:
        rmtree('dataset/')
    except BaseException:
        pass  # directory doesn't yet exist, no need to clear it
    os.makedirs("dataset/")

    pr_train_tweets = process_tweets(train_tweets)
    pr_test_tweets = process_tweets(test_tweets)

    #write the resulted sets to tsv files
    write_tsv(pr_train_tweets, "./dataset/train_set.tsv")
    write_tsv(pr_test_tweets, "./dataset/test_set.tsv")

    print("Final dataset consists of:")
    print("Train set:", len(pr_train_tweets))
    print("Test set:", len(pr_test_tweets))

if __name__ == "__main__":
	main()