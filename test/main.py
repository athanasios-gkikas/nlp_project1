import os
import model_test
import random

if __name__ == "__main__" :

    '''
    
    c = "I'm a Barbie girl in a Barbie's world. Life in plastic is fantastic. " \
           "You can brush my hair and touch me everywhere. " \
           "Imagination, life is your creation! " \
           "Com' on Barbie, let's go partying! "

    sentectesArr = []
    sentences = nltk.tokenize.sent_tokenize(c)
    for sentence in sentences :
        sentence = sentence.lower()
        translator = str.maketrans('', '', ".,!?)(-")
        sentence = sentence.translate(translator)
        #translator = str.maketrans('\'', '_')
        #sentence = sentence.translate(translator)
        sentectesArr += (nltk.word_tokenize(sentence),)

    #lfTokens = low_freq_tokens(sentectesArr, 10)
    #sentectesArr = remove_tokens(sentectesArr, lfTokens, "*UNK*")

    st = list()
    for i in range(1, 3) :
        st += [str('*start' + str(i) + '*'), ]

    sequence = list()

    for sentence in sentectesArr :
        sequence.extend(st + sentence + ['*end*',])

    trigrams = get_ngram(sequence, 3)
    for (i,gram) in enumerate(trigrams):
        print('Gram {0}: {1}'.format(i,gram))
    '''
    '''
    for s in sentectesArr :
        s = st + s + ['*end*',]
        print(s)
        trigrams = nltk.ngrams(s, 3)
        #trigrams = get_ngram(s, 3)
        for (i,gram) in enumerate(trigrams):
            print('Gram {0}: {1}'.format(i,gram))
    '''
    '''
    '''

    cwd = os.getcwd()
    root = cwd + "//corpus//"
    corpus = root + "europarl.en"

    m = model_test.model(root)
    # m.build(corpus)
    m.load_counts()
    queries = m.import_queries()
    # print(queries[:2])
    print(len(m.import_queries()))
    print("Bigram: ", m.get_language_cross_entropy(queries, 2))
    print("Trigram: ", m.get_language_cross_entropy(queries, 3))
    for l in range(0, 105, 5):
        print("l:", l/100., " LANG CE: ", m.get_interpolated_language_cross_entropy(queries, l/100.))
    # probs1 = m.log_probs(queries, 2)
    # for q in queries : random.shuffle(q)
    # probs2 = m.log_probs(queries, 2)
    # print(probs1)
    # print(probs2)
