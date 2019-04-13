from nltk import ngrams
from random import shuffle

import pyconll
import os
import gzip
import json
import numpy as np

def load_data(pData) :

	pos_data = []
	for sentence in pData:
		token_list = []
		for token in sentence:
			token_list.append([token.form, token.upos
				if token.upos is not None else "None"])

		pos_data.append(token_list)

	return pos_data

def buildDataset(pData, pNgram) :

	dataset = []

	for sentence in pData :
		token_list = []
		for i in range(0, int(pNgram / 2)) :
			token_list.append(["__PAD__", "__PAD__"])

		for token in sentence :
			token_list.append(token)

		for i in range(0, int(pNgram / 2)) :
			token_list.append(["__PAD__", "__PAD__"])

		for i, ngram in enumerate(ngrams(token_list, pNgram)) :
			gram = []
			for token in ngram :
				gram.append(token[0])
			dataset.append([gram, ngram[int(pNgram / 2)][1]])

	return dataset

def export_arr(pFile, pData) :

	with gzip.GzipFile(pFile + ".json.gz", "wb") as file :
		file.write(json.dumps(pData, indent = 4).encode('utf-8'))

	return

def import_arr(pFile) :

	with gzip.GzipFile(pFile, "rb") as file :
		data = json.loads(file.read().decode('utf-8'))

	return data

def importCorpus() :
	cwd = os.getcwd()
	db = import_arr(cwd + "/dataset/5gram_database.json.gz")
	return db[0], db[1], db[2], db[3]


def export_embeddings(pX, pY, pFile) :
    cwd = os.getcwd()
    np.savez_compressed(cwd + "/dataset/" + pFile, a=pX, b=pY)
    return

def import_embeddings(pFile) :
    cwd = os.getcwd()
    data = np.load(cwd + "/dataset/" + pFile + ".npz")
    return data['a'], data['b']


def exportCorpus(pWindow = 5):

	cwd = os.getcwd()

	train_path = cwd + "/dataset/en_partut-ud-train.conllu"
	dev_path = cwd + "/dataset/en_partut-ud-dev.conllu"
	test_path = cwd + "/dataset/en_partut-ud-test.conllu"

	train_data = pyconll.load_from_file(train_path)
	dev_data = pyconll.load_from_file(dev_path)
	test_data = pyconll.load_from_file(test_path)

	train = load_data(train_data)
	dev = load_data(dev_data)
	test = load_data(test_data)

	shuffle(train)
	sz = int(len(train) * 0.1)
	val = train[:sz]
	train = train[sz:]

	train = buildDataset(train, pWindow)
	val = buildDataset(val, pWindow)
	dev = buildDataset(dev, pWindow)
	test = buildDataset(test, pWindow)

	export_arr(cwd + "/dataset/5gram_database", (train, val, dev, test))

	return