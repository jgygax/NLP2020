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

print("cleaned")
# %%
ps = PorterStemmer()

wordfreq = {}
for article in texts:
    tokens = nltk.word_tokenize(article)
    for token in tokens:
        token = ps.stem(token)
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1
# %%
wordfreq_wo_stemming = {}
for article in texts:
    tokens = nltk.word_tokenize(article)
    for token in tokens:
        if token not in wordfreq_wo_stemming.keys():
            wordfreq_wo_stemming[token] = 1
        else:
            wordfreq_wo_stemming[token] += 1

print("stemmed")
# %%
most_freq_100 = heapq.nlargest(100, wordfreq, key=wordfreq.get)
most_freq_1000 = heapq.nlargest(1000, wordfreq, key=wordfreq.get)
most_freq_10000 = heapq.nlargest(10000, wordfreq, key=wordfreq.get)
all_stemming = wordfreq
all_wo_stemming = wordfreq_wo_stemming

sizes = [most_freq_100, most_freq_1000,
         most_freq_10000]

# %%
all_vectors = []
for size in sizes:
    article_vectors = []
    for article in texts:
        article_tokens = nltk.word_tokenize(article)
        article_vec = []
        for token in size:
            if token in article_tokens:
                article_vec.append(1)
            else:
                article_vec.append(0)
        article_vectors.append(article_vec)
    article_vectors = torch.FloatTensor(article_vectors)
    all_vectors.append(article_vectors)

print("vectorized")
# %%

with open('onehot-times-all.npy', 'wb') as f:
    np.save(f, all_vectors[0])
    np.save(f, all_vectors[1])
    np.save(f, all_vectors[2])
    np.save(f, np.asarray(labels))

print("saved")
# %%

size_num = [100,1000,10000]

ref_list = list(set(labels))
cleaned_labels = [ref_list.index(i) for i in labels]

# %%
for idx, data in enumerate(all_vectors):

    class LogLinearModel(torch.nn.Module):
        def __init__(self):
            super(LogLinearModel, self).__init__()
            self.linear = torch.nn.Linear(size_num[idx], 28)

        def forward(self, x):
            y_pred = F.sigmoid(self.linear(x))
            return y_pred

    print("--------------------------------------------------")
    print("Using a vocabulary of: ", sizes[idx])
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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(500):
        model.train()
        optimizer.zero_grad()                       # Forward pass
        y_pred = model(train_data)                  # Compute Loss
        loss = criterion(y_pred, train_labels)      # Backward pass
        loss.backward()
        optimizer.step()



    y_pred = model(test_data)
    pred=torch.max(y_pred.data,1)

    pred = pred[1].numpy()
    print("================================================")
    print("   USING BAG OF WORDS AND LOGISTIC REGRESSION   ")
    print("================================================")

    print("Accuracy: \t", accuracy_score(test_labels, pred))
    print("F-score: \t", f1_score(test_labels, pred, average='macro'))
