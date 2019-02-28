import nltk
import os
import string
import random
import json
import math

class model :

    def __init__(self, pRoot = None) :

        self.mRoot = pRoot if pRoot is not None else os.getcwd() + "//"
        self.mUnigramCounts = None
        self.mBigramCounts = None
        self.mTrigramCounts = None

        return

    def build(self, pFile) :

        self.export_corpus(pFile)

        unigram = self.build_ngrams(self.mRoot + "train_corpus.txt", 1)
        bigram = self.build_ngrams(self.mRoot + "train_corpus.txt", 2)
        trigram = self.build_ngrams(self.mRoot + "train_corpus.txt", 3)

        self.mUnigramCounts = self.compute_counts(unigram)
        self.mBigramCounts = self.compute_counts(bigram)
        self.mTrigramCounts = self.compute_counts(trigram)

        self.export_arr(self.mRoot + "unigram.txt", [(k,v) for (k,v) in self.mUnigramCounts.items()])
        self.export_arr(self.mRoot + "bigram.txt", [(k,v) for (k,v) in self.mBigramCounts.items()])
        self.export_arr(self.mRoot + "trigram.txt", [(k,v) for (k,v) in self.mTrigramCounts.items()])

        return

    def load_counts(self) :

        print("Loading unigrams")
        self.mUnigramCounts = {tuple(pair[0]) : pair[1] for pair in self.import_arr(
            self.mRoot + "unigram.txt")}

        print("Loading bigrams")
        self.mBigramCounts ={tuple(pair[0]) : pair[1] for pair in self.import_arr(
            self.mRoot + "bigram.txt")}

        print("Loading trigrams")
        self.mTrigramCounts = {tuple(pair[0]) : pair[1] for pair in self.import_arr(
            self.mRoot + "trigram.txt")}

        return

    def export_arr(self, pFile, pArray) :

        with open(pFile, mode="w", encoding='utf8') as file :
            json.dump(pArray, file)

        return

    def import_arr(self, pFile) :

        with open(pFile, mode="r", encoding='utf8') as file :
            lines = json.loads(file.read())

        return lines

    def low_freq_tokens(self, pSentences, pThresh) :

        voc = dict()

        for sentence in pSentences :
            for token in sentence :
                if token in voc :
                    voc[token] += 1
                else :
                    voc[token] = 1

        return {k for k, v in voc.items() if v < pThresh}

    def remove_tokens(self, pSentences, pTokens, pReplacement) :

        for sentence in pSentences :
            for i, token in enumerate(sentence) :
                if token in pTokens :
                    sentence[i] = pReplacement

        return pSentences

    def export_corpus(self, pFile) :

        punctuation = str.maketrans('', '', ".,!?)(-")
        replace = str.maketrans('\'', '_')

        sentectesArr = []

        with open(pFile, mode="r", encoding='utf8') as corpus_desrc :
            print("Reading corpus")
            corpus = corpus_desrc.read()

            print("Extracting sentences")
            sentences = nltk.tokenize.sent_tokenize(corpus)

            print("Processing sentences")
            for sentence in sentences :
                sentence = sentence.lower()
                sentence = sentence.translate(punctuation)
                #sentence = sentence.translate(replace)
                sentectesArr += (nltk.word_tokenize(sentence),)

        size = len(sentectesArr)
        train_sz = int(size * 0.7)
        val_sz = int(size * 0.2)

        train = sentectesArr[:train_sz]
        val = sentectesArr[train_sz:train_sz + val_sz]
        test = sentectesArr[train_sz + val_sz:]

        print("Extracting low frequency tokens")
        lfTokens = self.low_freq_tokens(train, 10)

        print("Removing low frequency tokens")
        train = self.remove_tokens(train, lfTokens, "*UNK*")
        val = self.remove_tokens(val, lfTokens, "*UNK*")
        test = self.remove_tokens(test, lfTokens, "*UNK*")

        print("Exporting dataset")
        self.export_arr(self.mRoot + "train_corpus.txt", train)
        self.export_arr(self.mRoot + "val_corpus.txt", val)
        self.export_arr(self.mRoot + "test_corpus.txt", test)

        return

    def get_ngram(self, pTokens, pN) :

        ngram = [[None] * pN] * (len(pTokens) - pN + 1)

        for i in range(len(pTokens) - pN + 1) :
            ngram[i] = pTokens[i : i + pN]

        return ngram

    def build_ngrams(self, pFile, pN) :

        sentences = self.import_arr(pFile)

        st = list()

        for i in range(0, pN) :
            st += [str('*start' + str(i + 1) + '*'), ]

        sequence = list()

        print("Building sequence")
        for sentence in sentences :
            sequence.extend((st + sentence + ['*end*',]))

        print("Creating ngram ", pN)
        return self.get_ngram(sequence, pN)

    def compute_counts(self, pNgrams) :

        counts = {}

        print("Creating ngram map ", len(pNgrams[0]))

        for ngram in pNgrams :
            ngram = tuple(ngram)

            if ngram in counts :
                counts[ngram] += 1
            else :
                counts[ngram] = 1

        return counts

    def import_queries(self) :
        return self.import_arr(self.mRoot + "test_corpus.txt")

    def log_probs(self, pQueries, pN, pMerge = False) :

        if self.mUnigramCounts is None or (
           self.mBigramCounts is None) or (
           self.mTrigramCounts is None) : return

        st = list()

        for i in range(1, pN) :
            st += [str('*start' + str(i) + '*'), ]

        sequences = list()

        print("Processing queries")
        if pMerge :
            for sentence in pQueries :
                sequences.extend((st + sentence + ['*end*',]))
            sequences = [sequences,]
        else :
            for sentence in pQueries :
                sequences += (st + sentence + ['*end*',],)

        probs = [0.0] * len(sequences)
        ngramList = (self.mUnigramCounts, self.mBigramCounts, self.mTrigramCounts)

        print("Computing probs")
        for i, seq in enumerate(sequences) :
            gram = self.get_ngram(seq, pN)

            for g in gram :
                cEnum = 0
                cDenom = 0
                V = len(ngramList[0]) - 1
                a = 1.0
                g = tuple(g)

                if g in ngramList[pN - 1] :
                    cEnum = ngramList[pN - 1][g]

                if pN > 1 :
                    if g[:pN - 1] in ngramList[pN - 2] :
                        cDenom = ngramList[pN - 2][g[:pN - 1]]
                else :
                    cDenom = len(ngramList[0])

                probs[i] += (math.log2(cEnum + a) - math.log2(cDenom + a * V))

        return probs

    def get_count(self, ngram):
        ngramList = (self.mUnigramCounts, self.mBigramCounts, self.mTrigramCounts)
        pN = len(ngram)
        if ngram in ngramList[pN-1]:
            count = ngramList[pN-1][ngram]
        else:
            count = 0
        return count

    def get_language_cross_entropy(self, pCorpus, pN) :
        sum_of_entropy = 0
        st = list()

        for i in range(1, pN) :
            st += [str('*start' + str(i) + '*'), ]

        sequences = list()

        for sentence in pCorpus :
            sequences.extend((st + sentence + ['*end*',]))
        sequences = [sequences,]

        ngrams = self.get_ngram(sequences[0], pN)
        ngramList = (self.mUnigramCounts, self.mBigramCounts, self.mTrigramCounts)

        for ngram in ngrams:
            ngram = tuple(ngram)
            if (ngram[-1] != "*start1*") and (ngram[-1] != "*start2*") :
                cDenom = 0.0
                V = len(ngramList[0]) - 1

                if pN > 1 :
                    if ngram[:pN - 1] in ngramList[pN - 2] :
                        cDenom = ngramList[pN - 2][ngram[:pN - 1]]

                if ngram in ngramList[pN - 1] :
                    count = ngramList[pN - 1][ngram]
                else :
                    count = 0

                temp_token_probability = (count + 1.) / (cDenom + V)
                sum_of_entropy += - math.log2(temp_token_probability)

        return sum_of_entropy / len(ngrams)

    def get_perplexity(self, pCorpus, pN) :
        sum_of_entropy = 0
        st = list()

        for i in range(1, pN) :
            st += [str('*start' + str(i) + '*'), ]

        sequences = list()

        for sentence in pCorpus :
            sequences.extend((st + sentence + ['*end*',]))
        sequences = [sequences,]

        ngrams = self.get_ngram(sequences[0], pN)
        ngramList = (self.mUnigramCounts, self.mBigramCounts, self.mTrigramCounts)

        for ngram in ngrams:
            ngram = tuple(ngram)
            if (ngram[-1] != "*start1*") and (ngram[-1] != "*start2*") :
                cDenom = 0.0
                V = len(ngramList[0]) - 1

                if pN > 1 :
                    if ngram[:pN - 1] in ngramList[pN - 2] :
                        cDenom = ngramList[pN - 2][ngram[:pN - 1]]

                if ngram in ngramList[pN - 1] :
                    count = ngramList[pN - 1][ngram]
                else :
                    count = 0

                temp_token_probability = (count + 1.) / (cDenom + V)
                sum_of_entropy += - math.log2(temp_token_probability)

        return pow(2, sum_of_entropy / len(ngrams))

    def get_interpolated_language_cross_entropy(self, pCorpus, l=0.5) :
        sum_of_entropy = 0
        st = list()

        for i in range(1, 3) :
            st += [str('*start' + str(i) + '*'), ]

        sequences = list()

        for sentence in pCorpus :
            sequences.extend((st + sentence + ['*end*',]))
        sequences = [sequences,]

        trigrams = self.get_ngram(sequences[0], 3)
        bigrams = self.get_ngram(sequences[0], 2)
        ngramList = (self.mUnigramCounts, self.mBigramCounts, self.mTrigramCounts)

        for ngram in trigrams:
            ngram = tuple(ngram)
            # if (ngram[-1] != "*start1*") and (ngram[-1] != "*start2*"):                                                   # Ion asks for it only in perplexity
            cDenom = 0.0
            V = len(ngramList[0]) - 1

######################################################################
            if ngram[:3 - 1] in ngramList[3 - 2]:
                cDenom = ngramList[3 - 2][ngram[:3 - 1]]
            else:
                cDenom = 0

            if ngram in ngramList[3 - 1]:
                count = ngramList[3 - 1][ngram]
            else:
                count = 0

            trigram_probability = (count + 1.) / (cDenom + V)  # Laplace smoothing
            if (trigram_probability < 0.) or (trigram_probability > 1.):
                print("YOU SUCK TRI!")
            # print("count:", count, " cDenom: ", cDenom, " V: ", V)
########################################################################

            if ngram[1] in ngramList[2 - 2] :
                cDenom = ngramList[2 - 2][ngram[1]]
            else :
                cDenom = 0

            if ngram[1:3] in ngramList[2 - 1] :
                count = ngramList[2 - 1][ngram[1:3]]
            else :
                count = 0

            bigram_probability = (count + 1.) / (cDenom + V)    # Laplace smoothing
            if (bigram_probability < 0.) or (bigram_probability > 1.):
                print("YOU SUCK BI!")
            # print("count:", count, " cDenom: ", cDenom, " V: ", V)
#########################################################################
            temp_token_probability = l*bigram_probability + (1-l)*trigram_probability
            # print(bigram_probability)
            # print(trigram_probability)
            # print(temp_token_probability)
            sum_of_entropy += - math.log2(temp_token_probability)

        return sum_of_entropy / len(trigrams)
