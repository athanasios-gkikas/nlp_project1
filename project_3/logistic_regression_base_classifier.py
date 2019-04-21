import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


def import_embeddings(filepath) :
    data = np.load(filepath)
    return data['a'], data['b']


# ========== LOADING DATA ============
cwd = os.getcwd()
training_filepath = cwd + "/dataset/train.npz"
x_train, y_train = import_embeddings(training_filepath)
# print(x_train.shape)
# print(y_train.shape)
# print(y_train)
y_train = [np.where(y==1)[0][0] for y in y_train]
# print(y_train)

# ====== TRAINING CLASSIFIER ========
lr_classifier = LogisticRegression(random_state=42,
                                   solver='lbfgs',                                         # newton-cg, sag, saga, lbfgs
                                   multi_class='multinomial',
                                   verbose=1).fit(x_train, y_train)


# ====== EVALUATING CLASSIFIER ======
test_filepath = cwd + "/dataset/test.npz"
x_test, y_test_true = import_embeddings(test_filepath)
y_test_true = [np.where(y==1)[0][0] for y in y_test_true]
y_test_pred = lr_classifier.predict(x_test)

pos_dict = {0: "ADJ", 1: "ADP", 2: "ADV", 3: "AUX", 4: "CCONJ", 5: "DET", 6: "INTJ", 7: "NOUN", 8: "NUM", 9: "PART", 10: "PRON", 11: "PROPN", 12: "PUNCT", 13: "SCONJ", 14: "SYM", 16: "VERB", 17: "X", 15: "UNK"}
y_test_true = [pos_dict[y] for y in y_test_true]
y_test_pred = [pos_dict[y] for y in y_test_pred]

print('accuracy:', accuracy_score(y_test_true, y_test_pred))
print('macro-f1-score:', f1_score(y_test_true, y_test_pred, average='macro'))
print('micro-f1-score:', f1_score(y_test_true, y_test_pred, average='micro'))
print(classification_report(y_test_true, y_test_pred))
