from random import shuffle
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pyconll
import os
from nltk.tokenize import word_tokenize


def buildDataset(pData) :
  x_train = []
  y_train = []

  for sentence in pData :

    for token in sentence :
      x_train.append(token[0])
      y_train.append(token[1])

  return x_train, y_train


def load_data(pData) :
    pos_data = []
    for sentence in pData:
        token_list = []
        for token in sentence:
            token_list.append([token.form, token.upos if token.upos is not None else "None"])

        pos_data.append(token_list)

    return pos_data


def train_most_freq_pos_classifier(x_train, y_train):
    count_pos_per_word = dict()

    for i in range(0, len(x_train)):
        if x_train[i] not in count_pos_per_word:
            count_pos_per_word[x_train[i]] = dict()
        if y_train[i] not in count_pos_per_word[x_train[i]]:
            count_pos_per_word[x_train[i]][y_train[i]] = 0
        count_pos_per_word[x_train[i]][y_train[i]] += 1

    #   print(count_pos_per_word)

    for key_word in count_pos_per_word:
        temp_dict = count_pos_per_word[key_word]
        maximum = 0
        most_freq_tag = ""

        for key_tag in temp_dict:
            if temp_dict[key_tag] > maximum:
                maximum = temp_dict[key_tag]
                most_freq_tag = key_tag

        count_pos_per_word[key_word] = most_freq_tag

    #   print(count_pos_per_word)
    return count_pos_per_word


def predict_sentence(sentence):
    y_pred = []
    #sentence = word_tokenize(sentence)

    for token in sentence:
        print(token)
        if token in most_freq_pos_per_word:
            y_pred.append(most_freq_pos_per_word[token])
        else:
            y_pred.append('UNK')

    return y_pred

cwd = os.getcwd()
train_path = cwd + "/dataset/en_ewt-ud-train.conllu"
train_data = pyconll.load_from_file(train_path)
train = load_data(train_data)

x_train, y_train = buildDataset(train)
# print(x_train)
# print(y_train)

most_freq_pos_per_word = train_most_freq_pos_classifier(x_train, y_train)

test_path = cwd + "/dataset/en_ewt-ud-test.conllu"
test_data = pyconll.load_from_file(test_path)
test = load_data(test_data)
shuffle(test)
x_test, y_test_true = buildDataset(test)

y_test_pred = []

for word in x_test:
    if word in most_freq_pos_per_word:
        y_test_pred.append(most_freq_pos_per_word[word])
    else:
        y_test_pred.append('UNK')

# print(len(x_test))
# print(y_test_true)
# print(y_test_pred)

print('accuracy:', accuracy_score(y_test_true, y_test_pred))
print(classification_report(y_test_true, y_test_pred))

#print(predict_sentence("This is a nice car!"))
