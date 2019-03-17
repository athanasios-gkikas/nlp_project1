import os
import model
import random

def test_case_custom_corpus() :

    m = model.model()

    tokens = ("*start1*", "*start2*", "a", "b", "*end*",) * 10

    unigram = m.get_ngram(tokens, 1)
    bigram = m.get_ngram(tokens, 2)
    trigram = m.get_ngram(tokens, 3)
    voc = m.compute_counts([unigram, bigram, trigram])
    m.set_voc(voc, 3)

    query = [["a",],]

    print(m.probs(query, 3))
    print(m.log_probs(query, 3))

    return

def test_case_full_corpus() :

    cwd = os.getcwd()
    root = cwd + "//dataset//"
    corpus = root + "europarl.en"

    m = model.model(root)
    #m.build(corpus) # do not enable this
    m.load_counts()
    #print("Tuning result: ", m.tune())

    queries = [
        ['i', 'think', 'that', 'the', 'honourable', 'member', 'raises', 'an', 'important', 'point'],
        ['member', 'the', 'think', 'that', 'raises', 'i', 'important', 'an', 'honourable', 'point'],
        ['it', 'possesses', 'political', 'economic', 'and', 'diplomatic', 'leverage'],
        ['economic', 'it', 'leverage', 'and', 'diplomatic', 'possesses', 'political'],
    ]

    #queries = m.import_queries()
    #for q in queries : print(q)

    #print(m.get_kn_smoothing(queries[0], 10))

    #print("Perplexity Bigram: ", m.perplexity(queries, 2))
    #print("Unigram: ", m.language_cross_entropy(queries, 1))
    #print("Bigram: ", m.language_cross_entropy(queries, 2))
    #print("Bigram: ", m.perplexity(queries, 2))
    #print("Trigram: ", m.language_cross_entropy(queries, 3))
    #print("Trigram: ", m.perplexity(queries, 3))
    #print("Interpolated: ", m.language_cross_entropy(queries, 3))
    #print("Interpolated: ", m.perplexity(queries, 3))

    #print(m.log_probs(queries, 1))
    #print(m.log_probs(queries, 2))
    print(m.log_probs(queries, 3))
    #for q in queries :
    #   random.shuffle(q)
    #    print(q)

    #print(m.log_probs(queries, 2))

    #print(m.evaluate(queries))

    return

if __name__ == "__main__" :

    #test_case_custom_corpus()
    test_case_full_corpus()
