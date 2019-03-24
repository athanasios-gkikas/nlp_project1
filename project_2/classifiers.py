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
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC, SVC, NuSVR
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from time import time

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

def encodeSet(pSet) :

    x = [s[1] for s in list(pSet.values())]
    y = [s[0] for s in list(pSet.values())]
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    return x, y

def tuner(pTrain, pPipeline, pParams) :

    grid = GridSearchCV(pPipeline,
        param_grid=pParams, cv=5, n_jobs=-1, verbose=1)

    t0 = time()

    grid.fit(pTrain[0], pTrain[1])

    print("done in %0.3fs" % (time() - t0))

    print("Best score: %0.3f" % grid.best_score_)
    print("Best parameters set:")
    best_parameters = grid.best_estimator_.get_params()

    for param_name in sorted(pParams.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    return grid.best_estimator_.named_steps['clf']

def tuneSVC(pTrain) :

    pipeline = Pipeline([
        ('clf',  OneVsRestClassifier(SVC(
            class_weight="balanced",
            max_iter=1000,
            gamma='scale',
            random_state=0))),])

    parameters = {
        'clf__estimator__kernel' : ['linear', 'rbf', 'poly'],
        'clf__estimator__C' : [0.01, 0.1, 1.0, 10.0, 100.0],}

    return tuner(pTrain, pipeline, parameters)

def tuneSGD(pTrain) :

    pipeline = Pipeline([
        ('clf', OneVsRestClassifier(SGDClassifier(
            max_iter=10000,
            tol=1.e-4,
            n_jobs=-1,
            class_weight='balanced',
            random_state=0))),])

    parameters = {
        'clf__estimator__penalty': ['l2', 'elasticnet'],
        'clf__estimator__l1_ratio': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'clf__estimator__loss': ['modified_huber', 'perceptron'],}

    return tuner(pTrain, pipeline, parameters)

def tuneBase(pTrain) :

    pipeline = Pipeline([
        ('clf',  DummyClassifier(strategy = "most_frequent", random_state=0)),])

    parameters = { 'clf__strategy': ['most_frequent', 'stratified'], }

    return tuner(pTrain, pipeline, parameters)

def tuneLogistic(pTrain) :

    pipeline = Pipeline([
        ('clf',  LogisticRegression(
            class_weight="balanced",
            max_iter=10000,
            n_jobs=-1,
            random_state=0,
            multi_class='multinomial',
            penalty='l2')),])

    parameters = {
        'clf__solver': ['newton-cg', 'lbfgs', 'sag',],
        'clf__C' : [0.01, 0.1, 1.0, 10.0,]}

    return tuner(pTrain, pipeline, parameters)

def approxDim(pData) :

    maxFeatures = tuple()
    #(700, 1225, 1500, 2000, 2200, 2275, 2650, 3025, 2850, 3162, 3475, 3787, 3700)

    for f in range(1000, 14000, 1000) :
        print("num features: ", f)

        vectorizer = TfidfVectorizer(
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
            norm='l2',
            ngram_range=(1,3),
            max_features=f)

        tfidf = vectorizer.fit_transform(pData)

        svd = TruncatedSVD(n_components=1, n_iter=1, random_state=0)
        left = 100
        right = f - 100

        while True :
            c = (left + right) // 2
            svd = TruncatedSVD(n_components=c, n_iter=1, random_state=0)
            svd.fit(tfidf)
            var = svd.explained_variance_ratio_.sum()
            if var > 0.88 and var < 0.93 :
                print(c)
                maxFeatures += (c,)
                break
            elif var > 0.93 :
                right = c
            else :
                left = c

    print(maxFeatures)
    arr = np.array(maxFeatures).transpose()
    x = np.arange(1000, 14000, 1000)

    fig, ax = plt.subplots()
    ax.plot(x, arr)
    ax.set(xlabel='initial domain', ylabel='projected domain')
    ax.grid()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x, arr / x)
    ax.set(xlabel='ratio', ylabel='domain')
    ax.grid()
    plt.show()

    arr = np.log(arr)

    return int(np.exp(arr.sum() / len(arr)))

def main():

    cwd = os.getcwd()
    train_tweets = load_tsv(cwd + "/project_2/dataset/train_set.tsv")
    test_tweets = load_tsv(cwd + "/project_2/dataset/test_set.tsv")

    pr_train_tweets = process_tweets(train_tweets)
    pr_test_tweets = process_tweets(test_tweets)

    x_train, y_train = encodeSet(pr_train_tweets)
    x_test, y_test = encodeSet(pr_test_tweets)

    f = 2281 #approxDim(y_train)
    vectorizer = TfidfVectorizer(
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
        norm='l2',
        ngram_range=(1,3),
        max_features=f)

    vectorizer.fit(x_train)
    x_train = vectorizer.transform(x_train)
    x_test = vectorizer.transform(x_test)

    #model = tuneBase([x_train, y_train])
    model = tuneSVC([x_train, y_train])
    #model = tuneSGD([x_train, y_train])
    #model = tuneLogistic([x_train, y_train])

    plot_learning_curve(model,
        "Learning Curves for Logistic Regression",
        x_train, y_train, (0.1, 1.01), cv=5,
        n_jobs=-1).show()

    plot_precision_recall_curves(model,
        "Logistic Regression",
        x_train, y_train, x_test, y_test)

if __name__ == "__main__":
	main()
