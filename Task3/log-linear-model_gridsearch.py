# Julia Gygax
# jgygax@ethz.ch
# 16-922-064

# %%
import heapq
import nltk
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
import random
import string

import re

import os
import random
import csv
from sklearn.metrics import accuracy_score, f1_score

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

import torch
from torch.autograd import Variable
from torch.nn import functional as F
import string
from nltk.corpus import stopwords

import itertools
# %%
# read data
data = pd.read_csv(
    'boydstun_nyt_frontpage_dataset_1996-2006_0_pap2014_recoding_updated2018.csv',  encoding="ISO-8859-1")
# %%
# extract texts and labels
texts = []
labels = []

for i, row in data.iterrows():
    labels.append(row.majortopic)
    check = row.isnull()
    if check[-2] and check[-1]:
        texts.append(' ')
    elif check[-2]:
        texts.append(row.summary)
    elif check[-1]:
        texts.append(row.title)
    else:
        texts.append(row.title + ' ' + row.summary)

print("extracted")
# %%
for i in range(len(texts)):
    texts[i] = texts[i].lower()
    texts[i] = re.sub(r'\W', ' ', texts[i])
    texts[i] = re.sub(r'\s+', ' ', texts[i])

ps = PorterStemmer()

cleaned_texts = []
for text in texts:
    # split into words
    tokens = nltk.word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    stemmed = [ps.stem(word) for word in words]
    cleaned_texts.append(stemmed)

print("cleaned")
# %%

wordfreq = {}
for article in texts:
    tokens = nltk.word_tokenize(article)
    for token in tokens:
        token = ps.stem(token)
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1

print("frequencies")
# %%
most_freq_100 = heapq.nlargest(100, wordfreq, key=wordfreq.get)
most_freq_1000 = heapq.nlargest(1000, wordfreq, key=wordfreq.get)
most_freq_10000 = heapq.nlargest(10000, wordfreq, key=wordfreq.get)
all_stemming = wordfreq

sizes = [list(most_freq_100), list(most_freq_1000),
         list(most_freq_10000)]

# %%
all_vectors = []
for size in sizes:
    article_vectors = []
    for article_tokens in cleaned_texts:
        article_vec = [0]*len(size)
        for token in article_tokens:
            if token in size:
                article_vec[size.index(token)] += 1
        article_vectors.append(article_vec)
    article_vectors = torch.FloatTensor(article_vectors)
    all_vectors.append(article_vectors)

print("vectorized")
# %%

# with open('onehot-times-all.npy', 'wb') as f:
#     np.save(f, all_vectors[0])
#     np.save(f, all_vectors[1])
#     np.save(f, all_vectors[2])
#     np.save(f, np.asarray(labels))

print("saved")
# %%

size_num = [100, 1000, 10000, len(all_stemming)]

ref_list = list(set(labels))
cleaned_labels = [ref_list.index(i) for i in labels]

# %%

lrs=  [0.05]
max_epochs=[50]
weight_decays = [0.001,0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1, 10, 12, 20, 100]

idx=2
data=all_vectors[idx]


class LogLinearModel(torch.nn.Module):
    def __init__(self):
        super(LogLinearModel, self).__init__()
        self.linear = torch.nn.Linear(size_num[idx], 28)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


max_accuracy = 0
m_F = 0
acc_test = 0
f_test = 0
params = []
for lr, max_epoch, weight_decay in itertools.product(lrs, max_epochs, weight_decays):
    print("--------------------------------------------------")
    print("Using a vocabulary of: ", size_num[idx])
    n = len(data)
    train_len = int(0.6*n)
    dev_len = int(0.8*n)

    train_data = data[0:train_len]
    dev_data = data[train_len:dev_len]
    test_data = data[dev_len:]
    train_labels = torch.LongTensor(cleaned_labels[0:train_len])
    dev_labels = torch.LongTensor(cleaned_labels[train_len:dev_len])
    test_labels = torch.LongTensor(cleaned_labels[dev_len:])

    model = LogLinearModel()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(max_epoch):
        model.train()
        optimizer.zero_grad()                       # Forward pass
        y_pred = model(train_data)                  # Compute Loss
        loss = criterion(y_pred, train_labels)      # Backward pass
        loss.backward()
        optimizer.step()

    y_pred_val = model(dev_data)
    pred_val = torch.max(y_pred_val.data, 1)

    pred_val = pred_val[1].numpy()

    print("================================================")
    print(lr, max_epoch, weight_decay)
    print("================================================")
    acc =  accuracy_score(dev_labels, pred_val)
    f = f1_score(dev_labels, pred_val, average='macro')
    if max_accuracy < acc:
        max_accuracy = acc
        m_f = f
        params = [lr, max_epoch, weight_decay]

        y_pred_test = model(test_data)
        pred_test = torch.max(y_pred_test.data, 1)

        pred_test = pred_test[1].numpy()
        acc_test = accuracy_score(test_labels, pred_test)
        f_test = f1_score(test_labels, pred_test, average='macro')

    print("Accuracy: \t", acc)
    print("F-score: \t", f)


print(max_accuracy, m_f, acc_test, f_test, *params)
# %%

# For size 100:  0.05 150 0.001