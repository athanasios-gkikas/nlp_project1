import gzip
import glob
import json
import math
import nltk
import os
import random
import string
import sys

class model :

    def __init__(self, pRoot = None) :

        self.mRoot = pRoot if pRoot is not None else os.getcwd() + "//"
        self.mNgramList = [[] for _ in range(3)]
        self.mLambda = 1.0

        return

    def build(self, pFile, pShuffleSentences = False) :

        self.export_corpus(pFile, pShuffleSentences)

        unigram = self.build_ngrams(self.mRoot + "corpus//train_corpus", 1)
        bigram = self.build_ngrams(self.mRoot + "corpus//train_corpus", 2)
        trigram = self.build_ngrams(self.mRoot + "corpus//train_corpus", 3)

        self.mNgramList[0] = self.compute_counts(unigram)
        self.mNgramList[1] = self.compute_counts(bigram)
        self.mNgramList[2] = self.compute_counts(trigram)

        print("Exporting ngrams")
        self.export_lists(self.mRoot + "unigrams//unigram", self.mNgramList[0])
        self.export_lists(self.mRoot + "bigrams//bigram", self.mNgramList[1])
        self.export_lists(self.mRoot + "trigrams//trigram", self.mNgramList[2])

        return

    def load_counts(self) :

        print("Loading unigrams")
        self.load_ngram(self.mRoot + "unigrams//*", 0)

        print("Loading bigrams")
        self.load_ngram(self.mRoot + "bigrams//*", 1)

        print("Loading trigrams")
        self.load_ngram(self.mRoot + "trigrams//*", 2)

        return

    def set_voc(self, pDic, pModel) :
        self.mNgramList[pModel - 1] = pDic
        return

    def load_ngram(self, pFile, pIdx) :

        for i, source in enumerate(glob.glob(pFile)) :
            self.mNgramList[pIdx] += [{
                tuple(pair[0]) : pair[1] for pair in self.import_arr(source) }]

        return

    def export_lists(self, pFile, pLists) :

        for i, ngrams in enumerate(pLists) :
            self.export_arr(
                pFile + "_" + str(i),
                [(k,v) for k, v in ngrams.items()])

        return

    def export_arr(self, pFile, pArray) :

        with gzip.GzipFile(pFile + ".json.gz", "wb") as file :
            file.write(json.dumps(pArray, indent = 4).encode('utf-8'))

        return

    def import_arr(self, pFile) :

        with gzip.GzipFile(pFile, "rb") as file :
            lines = json.loads(file.read().decode('utf-8'))

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

    def export_corpus(self, pFile, pShuffleSentences) :

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

        if pShuffleSentences :
            random.shuffle(sentectesArr)

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
        self.export_arr(self.mRoot + "corpus//train_corpus", train)
        self.export_arr(self.mRoot + "corpus//val_corpus", val)
        self.export_arr(self.mRoot + "corpus//test_corpus", test)

        return

    def get_ngram(self, pTokens, pN) :

        ngram = [[None] * pN for _ in range (len(pTokens) - pN + 1)]

        for i in range(len(pTokens) - pN + 1) :
            ngram[i] = pTokens[i : i + pN]

        return ngram

    def augment_seq(self, pSeq, pN, pMerge = False) :

        st = list()

        if pN == 1 :
            st += ['*start*',]
        else :
            for i in range(1, pN) :
                st += [str('*start' + str(i) + '*'), ]

        sequences = list()

        if pMerge :
            for sentence in pSeq :
                sequences.extend((st + sentence + ['*end*',]))
            sequences = [sequences,]
        else :
            for sentence in pSeq :
                sequences += [st + sentence + ['*end*',],]

        return sequences

    def build_ngrams(self, pFile, pN) :

        sentences = self.import_arr(pFile + ".json.gz")

        print("Building sequence")
        sequence = self.augment_seq(sentences, pN, True)

        print("Creating ngram ", pN)
        return [self.get_ngram(sequence[0], n + 1) for n in range(0, pN)]

    def compute_counts(self, pNgramLists) :

        counts = list()

        print("Creating ngram maps")

        for i, ngramList in enumerate(pNgramLists) :
            voc = {}
            for ngram in ngramList :
                ngram = tuple(ngram)

                if ngram in voc :
                    voc[ngram] += 1
                else :
                    voc[ngram] = 1
            counts.append(voc)

        return counts

    def import_queries(self) :
        return self.import_arr(self.mRoot + "corpus//test_corpus.json.gz")

    def import_val(self) :
        return self.import_arr(self.mRoot + "corpus//val_corpus.json.gz")

    def get_voc_sz(self, pN) :
        return len(self.mNgramList[pN][0])

    def get_count(self, pNgram, pModel) :
        count = 0
        n = len(pNgram) - 1

        if pNgram in self.mNgramList[pModel - 1][n] :
            count = self.mNgramList[pModel - 1][n][pNgram]

        return count

    def accum_ngrams(self, pN) :
        sum = 0

        for ngram in self.mNgramList[pN - 1][0].values() :
            sum += ngram

        return sum

    def laplace_smoothing(self, pEnum, pDenom, pN) :
        V = self.get_voc_sz(pN - 1) - 1
        a = 1.0
        p = ((pEnum + a) / (pDenom + a * V))

        if p > 1.0 or p < 0.0 :
            print(
                "Prob out of range: ",
                " cEnum: ", pEnum,
                " cDenom: ", pDenom,
                " V: ", V)

        return p

    def prob(self, pNgram, pModel) :
        n = len(pNgram)

        cEnum = self.get_count(pNgram, pModel)
        cDenom = self.get_count(pNgram[:n - 1], pModel) if n > 1 else \
            self.accum_ngrams(pModel)

        #return self.laplace_smoothing(cEnum, cDenom, pModel)
        return self.kn_smoothing(pNgram, cEnum, cDenom, pModel, 0.75)

    def probs(self, pQueries, pN, pMerge = False) :
        print("Processing queries")
        sequences = self.augment_seq(pQueries, pN, pMerge)

        probs = [1.0 for _ in range(len(sequences))]

        print("Computing probs")
        for i, seq in enumerate(sequences) :
            gram = self.get_ngram(seq, pN)

            for g in gram :
                probs[i] *= self.prob(tuple(g), len(g))

        return probs

    def log_probs(self, pQueries, pN, pMerge = False) :
        print("Processing queries")
        sequences = self.augment_seq(pQueries, pN, pMerge)

        probs = [0.0 for _ in range(len(sequences))]

        print("Computing probs")
        for i, seq in enumerate(sequences) :
            grams = self.get_ngram(seq, pN)
            for gram in grams :
                p = math.log2(self.prob(tuple(gram), len(gram)))
                #print(gram, " ", p)
                probs[i] += p

        return probs

    def language_cross_entropy(self, pCorpus, pN) :
        sum_of_entropy = 0
        start_ngrams = 0
        sequences = self.augment_seq(pCorpus, pN, True)
        ngrams = self.get_ngram(sequences[0], pN)

        for token in sequences[0] :
            if (token == "*start1*") or \
               (token == "*start2*") or \
               (token == "*start*") :
               start_ngrams += 1

        for ngram in ngrams :
            if (ngram[-1] != "*start1*") and \
               (ngram[-1] != "*start2*") and \
               (ngram[-1] != "*start*") :
                sum_of_entropy += math.log2(self.prob(tuple(ngram), pN))
                #sum_of_entropy += self.interpolated_logProb(tuple(ngram))

        return -sum_of_entropy / (len(sequences[0]) - start_ngrams)

    def perplexity(self, pCorpus, pN) :
        return pow(2, self.language_cross_entropy(pCorpus, pN))

    def interpolated_logProb(self, pNgram) :
        return math.log2(
            self.mLambda * self.prob(pNgram, 3) +
            (1.0 - self.mLambda) * self.prob(pNgram[1:3], 3))

    def evaluate(self, pQueries, pMerge = False) :
        sequences = self.augment_seq(pQueries, 3, pMerge)

        probs = [0.0 for _ in range(len(sequences))]

        for i, seq in enumerate(sequences) :
            gram = self.get_ngram(seq, 3)
            for g in gram :
                probs[i] += self.interpolated_logProb(tuple(g))

        return probs

    def tune(self) :

        pCorpus = self.import_val()
        minCEntropy = [sys.float_info[0], 0, self.mLambda]

        start_ngrams = 0
        sequences = self.augment_seq(pCorpus, 3, True)
        trigrams = self.get_ngram(sequences[0], 3)

        for token in sequences[0] :
            if (token == "*start1*") or (token == "*start2*") :
                start_ngrams += 1

        for l in range(0, 102, 2) :
            sum_of_entropy = 0

            for ngram in trigrams:
                ngram = tuple(ngram)

                if (ngram[-1] != "*start1*") and \
                   (ngram[-1] != "*start2*") and \
                   (ngram[-1] != "*start*") :
                    trigram_prob = self.prob(ngram, 3)
                    bigram_prob = self.prob(ngram[1:3], 3)

                    sum_of_entropy += math.log2((l / 100.) * trigram_prob + \
                        (1.0 - (l / 100.)) * bigram_prob)

            cross_entropy = -sum_of_entropy / (len(sequences[0]) - start_ngrams)
            perplexity = pow(2, cross_entropy)

            #print("Lamda: ", l/100., " CE: ", cross_entropy)

            if cross_entropy < minCEntropy[0] :
                minCEntropy = [cross_entropy, perplexity, l / 100.]

        self.mLambda = minCEntropy[2]
        return minCEntropy

    def get_prev(self, wk, pModel) :
        prev_wk = 0
        for w in self.mNgramList[pModel - 1][0].keys() :
            temp = [w[0],]
            for k in wk : temp.extend([k,])
            #print("Prev ", temp)
            if self.get_count(tuple(temp), pModel) > 0:
                prev_wk += 1
        return prev_wk

    def get_next(self, wk, pModel, pCountNeg) :
        next_wk = 0
        for w in self.mNgramList[pModel - 1][0].keys() :
            temp = []
            for k in wk : temp.extend([k,])
            temp.extend([w[0],])
            #print("Next ", temp)
            if self.get_count(tuple(temp), pModel) > 0 and not pCountNeg:
                next_wk += 1

            if self.get_count(tuple(temp), pModel) == 0 and pCountNeg:
                next_wk += 1

        return next_wk

    def get_denom(self, w, pModel) :
        wk = 0
        for item in self.mNgramList[pModel - 1][pModel - 1].items() :
            if item[0][-2] == w :
                wk += item[1]
        return wk

    def kn_smoothing(self, pNgram, pEnum, pDenom, pModel, D):
        n = len(pNgram)

        if pEnum > 0:
            probability_ngram = (pEnum - D) / pDenom
        else:
            prev_wk = self.get_prev(pNgram[:n - 1], pModel)
            next_wk = self.get_next(pNgram[:n - 1], pModel, False)
            next_denom = self.get_next(pNgram[:n - 1], pModel, True)

            #print(pDenom, " ", next_denom)
            probability_ngram = (D * next_wk) / self.mNgramList[pModel - 1][0][pNgram[-2]]

        if probability_ngram > 1.0 or probability_ngram < 0.0 :
            print(pNgram[:n - 1], " ", next_wk, " ", prev_wk)
            print(pNgram, " ", probability_ngram, " cEnum ", pEnum, " cDenom ", pDenom)
            print(
                "Prob out of range: ",
                " cEnum: ", pEnum,
                " cDenom: ", pDenom)

        return probability_ngram