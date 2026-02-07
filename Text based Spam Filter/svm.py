# open the emails data
import random
import os
from liblinearutil import *
ham_dataset = os.listdir("dataset/ham")
spam_dataset = os.listdir("dataset/spam")
data = []

#load data
for get_path in ham_dataset:
    f = open("dataset/ham/" + get_path, "r")
    text_data = f.read()
    data.append([text_data, 1])

#load data
for get_path in spam_dataset:
    f = open("dataset/spam/" + get_path, "r")
    text_data = f.read()
    data.append([text_data, -1])

# shuffle the data and split them into training and testing data
random.shuffle(data)
train_data = data[0: int(len(data) / 2)]
test_data = data[int(len(data) / 2) + 1: -1]

# map tokens to number of occurrences
h_dict = {}
s_dict = {}
for d in train_data:
    if d[-1] == 1:
        for word in d[0].split():
            if word in h_dict:
                h_dict[word] += 1
            else:
                h_dict[word] = 1
    elif d[-1] == -1:
        for word in d[0].split():
            if word in s_dict:
                s_dict[word] += 1
            else:
                s_dict[word] = 1

# Take first most N recurring words in ham and spam individually and converge into single word lexicon
N = 1000
top_ham_words = sorted(h_dict, key=h_dict.get, reverse=True)[:N]
top_spam_words = sorted(s_dict, key=s_dict.get, reverse=True)[:N]

# Creating a dictionary that has unique words from ham & spam top words
word_dictionary = dict()
index = 0
for word in top_ham_words:
    if word not in word_dictionary:
        word_dictionary[word] = index
        index += 1
for word in top_spam_words:
    if word not in word_dictionary:
        word_dictionary[word] = index
        index += 1


# training
tst_dat = []
tst_classes = []
for d in train_data:
    vector = [0] * len(word_dictionary)
    for word in d[0].split():
        if word in word_dictionary:
            index = word_dictionary.get(word)
            vector[index] = 1
    tst_dat.append(vector)
    tst_classes.append(d[1])
y, x = tst_classes, tst_dat
prob = problem(y, x)
param = parameter()
m = train(prob, param)

# testing
tst_dat = []
tst_classes = []
for d in test_data:
    vector = [0] * len(word_dictionary)
    for word in d[0].split():
        if word in word_dictionary:
            index = word_dictionary.get(word)
            vector[index] = 1
    tst_dat.append(vector)
    tst_classes.append(d[1])
y, x = tst_classes, tst_dat
pred_labels, (ACC, MSE, SCC), pred_values = predict(y, x, m)
print(ACC)
