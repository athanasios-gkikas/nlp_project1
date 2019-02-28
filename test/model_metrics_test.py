from math import log

test_corpus = "I'm a Barbie girl in a Barbie's world. Life in plastic is fantastic. " \
           "You can brush my hair and touch me everywhere. " \
           "Imagination, life is your creation! " \
           "Com' on Barbie, let's go partying! "


def get_language_cross_entropy(model):
    list_of_ngrams = []
    list_of_ngrams = model.split_corpus_into_ngrams(test_corpus)

    sum_of_entropy = 0
    for ngram in list_of_ngrams:
        # if (ngram[-1] != "*start1*") and (ngram[-1] != "*start2*"):                                                   # Ion asks for it only in perplexity

        # temp_token_probability = model.smoothing(model.get_count(ngram[:]),
        #                                          model.get_count(ngram[:-1]))
        temp_token_probability = (model.get_count(ngram[:]) + 1) / (model.get_count(ngram[:-1] + model.get_vocabulary_size()))    # Laplace smoothing
        sum_of_entropy += - log(temp_token_probability, 2)

    language_cross_entropy = sum_of_entropy / len(list_of_ngrams)

    return language_cross_entropy


def get_perplexity(model):
    list_of_ngrams = []
    list_of_ngrams = model.split_corpus_into_ngrams(test_corpus)

    product_of_probabilities = 1
    for ngram in list_of_ngrams:
        if (ngram[-1] != "*start1*") and (ngram[-1] != "*start2*"):
            # temp_token_probability = model.smoothing(model.get_count(ngram[:]),
            #                                          model.get_count(ngram[:-1]))
            temp_token_probability = (model.get_count(ngram[:]) + 1) / (model.get_count(ngram[:-1] + model.get_vocabulary_size()))  # Laplace smoothing
            product_of_probabilities *= temp_token_probability

    perplexity = product_of_probabilities**(-1/float(len(list_of_ngrams)))

    return perplexity
