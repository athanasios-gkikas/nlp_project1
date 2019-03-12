import re
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords


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

def process_tweets(tweets):
	tweet_stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])
	new_tweets = {}

	print(my_punctuation)

	for tweet in tweets:
		text = tweets[tweet][1].lower() # convert text to lower-case
		text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text) # remove URLs
		text = re.sub('@[^\s]+', 'AT_USER', text) # remove usernames
		text = re.sub(r'#([^\s]+)', r'\1', text) # remove the # in #hashtag
		text = word_tokenize(text) # remove repeated characters (helloooooooo into hello)
		text = [word for word in text if word not in tweet_stopwords]

		new_tweets[tweet] = [tweets[tweet][0], text]

	return new_tweets


def main():

	train_tweets = load_tsv("./dataset/train_set.tsv")
	val_tweets = load_tsv("./dataset/val_set.tsv")
	dev_tweets = load_tsv("./dataset/dev_set.tsv")
	test_tweets = load_tsv("./dataset/test_set.tsv")

	pr_train_tweets = process_tweets(train_tweets)

	print(list(train_tweets.keys())[0])
	print(train_tweets[list(train_tweets.keys())[0]])

	print(list(pr_train_tweets.keys())[0])
	print(pr_train_tweets[list(pr_train_tweets.keys())[0]])


if __name__ == "__main__":
	main()