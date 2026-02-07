# get text based email
import os
import random

ham_dataset = os.listdir("dataset/ham")
spam_dataset = os.listdir("dataset/spam")
unique_train_data = dict()
data = []

# loading dataset file
for get_path in ham_dataset:
    file_path = open("dataset/ham/" + get_path, "r")
    text_data = file_path.read()
    data.append([text_data, "ham_email"])

for get_path in spam_dataset:
    file_path = open("dataset/spam/" + get_path, "r")
    text_data = file_path.read()
    data.append([text_data, "spam_email"])

# shuffled the dataset files, split training_data and testing_data
random.shuffle(data)
train_data = data[0: int(len(data) / 2)]
test_data = data[int(len(data) / 2) + 1: int(len(data))]

# map tokens to number of occurrence
h_dict = dict()
s_dict = dict()

# filtering ham and spam email
ham_list = list(filter(lambda x: x[-1] == 'ham_email', train_data))
spam_list = list(filter(lambda x: x[-1] == 'spam_email', train_data))

for words in ham_list:
    for item in words[0].split():
        if item in h_dict:
            h_dict[item] += 1
        else:
            h_dict[item] = 1
for words in spam_list:
    for item in words[0].split():
        if item in s_dict:
            s_dict[item] += 1
        else:
            s_dict[item] = 1

# program start
initially_ham_words_probability = 0.5
initially_spam_words_probability = 0.5
true_predict = 0
false_predict = 0
wrong_spam = 0
wrong_ham = 0
true_ham = 0
true_spam = 0

for l in test_data:
    posterrior_probability_ham = 1
    posterrior_probability_spam = 1
    for item in l[0].split():
        count_of_ham_words = h_dict[item] if item in h_dict else 0.000001
        count_of_spam_words = s_dict[item] if item in s_dict else 0.000001
        likilyhood_of_ham = count_of_ham_words / (count_of_ham_words + count_of_spam_words)
        likilyhood_of_spam = count_of_spam_words / (count_of_ham_words + count_of_spam_words)
        posterrior_probability_ham *= (likilyhood_of_ham * initially_ham_words_probability) / (
                    likilyhood_of_ham * initially_ham_words_probability + likilyhood_of_spam * initially_spam_words_probability)
        posterrior_probability_spam *= (likilyhood_of_spam * initially_spam_words_probability) / (
                    likilyhood_of_ham * initially_ham_words_probability + likilyhood_of_spam * initially_spam_words_probability)

    # check for true spam text
    if posterrior_probability_spam > posterrior_probability_ham and l[1] == "spam_email":
        true_predict += 1
        true_spam += 1
    # check for true ham text
    elif posterrior_probability_spam < posterrior_probability_ham and l[1] == "ham_email":
        true_predict += 1
        true_ham += 1
    # check false dataset for ham and spam text
    else:
        if l[1] == "spam_email":
            wrong_spam += 1
        if l[1] == "ham_email":
            wrong_ham += 1
        false_predict += 1

accuracy = (float(true_predict) / float(true_predict + false_predict)) * 100

print("True Predict", true_predict)
print("False Predict", false_predict)
print("Accuracy:- ", accuracy)

ham_when__it_is_spam = wrong_spam / (true_predict + false_predict)
spam_when_it_is_ham = wrong_ham / (true_predict + false_predict)
print(ham_when__it_is_spam)
print(spam_when_it_is_ham)
print(wrong_spam)
print(wrong_ham)
