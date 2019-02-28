import nltk
import string
from nltk.lm import Vocabulary

#TODO: find if europarl exists in nltk, rather than the below file
filepath = "C:\\Users\\gkikas\\Downloads\\el-en\\europarl-v7.el-en-test.en"

vocabulary = {}
tokens = []


def remove_punctuation(sentence):
    translator = str.maketrans('', '', string.punctuation)
    sentence = sentence.translate(translator)
    return sentence


def add_to_tokens(sentence):
    sentence_tokens = nltk.word_tokenize(sentence)
    tokens.extend(sentence_tokens)
    return sentence_tokens


def add_to_vocabulary(sentence_tokens):
    for token in sentence_tokens:
        if token in vocabulary.keys():
            vocabulary[token] += 1
        else:
            vocabulary[token] = 1


#TODO
def filter_OOV_vocabulary(vocabulary):
	new_vocabulary = {}
    for key in vocabulary.keys():
        if key != "*UNK*":
            if key in new_vocabulary.keys():
				new_vocabulary[key] += 1
			else:
				new_vocabulary[key] = 1
		else:
			

# initial_corpus = "I'm a Barbie girl in a Barbie's world. Life in plastic is fantastic. " \
#            "You can brush my hair and touch me everywhere. " \
#            "Imagination, life is your creation! " \
#            "Com' on Barbie, let's go partying! "
initial_corpus = ""

with open(filepath, encoding="utf8") as corpus_file:
    for line in corpus_file:
        sentences = nltk.tokenize.sent_tokenize(line)
        for sentence in sentences:
            sentence = remove_punctuation(sentence)
            sentence = " *start* " + sentence + " *end* "
            sentence_tokens = add_to_tokens(sentence)
            add_to_vocabulary(sentence_tokens)
    vocabulary = filter_OOV_vocabulary(vocabulary)
#
# vocabulary = Vocabulary(tokens, unk_cutoff=10, unk_label="*UNK*")                                                     # TODO: Vocabulary by NLTK not permitted

print(tokens)
print(vocabulary)


