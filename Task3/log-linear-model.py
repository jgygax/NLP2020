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
data = pd.read_csv('boydstun_nyt_frontpage_dataset_1996-2006_0_pap2014_recoding_updated2018.csv',  encoding = "ISO-8859-1")
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

# %%
for i in range(len(texts)):
    texts[i] = texts[i].lower()
    texts[i] = re.sub(r'\W', ' ', texts[i])
    texts[i] = re.sub(r'\s+', ' ', texts[i])

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
# %%
most_freq_100 = heapq.nlargest(100, wordfreq, key=wordfreq.get)
most_freq_1000 = heapq.nlargest(1000, wordfreq, key=wordfreq.get)
most_freq_10000 = heapq.nlargest(10000, wordfreq, key=wordfreq.get)
all_stemming = wordfreq
all_wo_stemming = wordfreq_wo_stemming

sizes = [most_freq_100, most_freq_1000, most_freq_10000, all_stemming, all_wo_stemming]

# %%
article_vectors_sizes = []
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
    article_vectors_sizes.append(article_vectors)
# %%
article_vectors = np.asarray(article_vectors)

with open('onehot10.npy', 'wb') as f:
    np.save(f, article_vectors)
    np.save(f, np.asarray(labels))
# %%
pca = PCA(n_components=200)
data = pca.fit_transform(article_vectors)

# %%

test_data = data[0:400]
train_data = data[400:]
test_labels = labels[0:400]
train_labels = labels[400:]
# %%

x_data = Variable(torch.Tensor([[10.0], [9.0], [3.0], [2.0]]))
y_data = Variable(torch.Tensor([[90.0], [80.0], [50.0], [30.0]]))
# %%


class LogLinearModel(torch.nn.Module):
    def __init__(self):
        super(LogLinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y = self.linear(x)
        y_pred = torch.sigmoid(y)
        return y_pred


# %%
model = LogLinearModel()
# %%
criterion = torch.nn.MSELoss(size_average=False)
# %%
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# %%
for epoch in range(20):
    model.train()
    optimizer.zero_grad()    # Forward pass
    y_pred = model(x_data)    # Compute Loss
    loss = criterion(y_pred, y_data)    # Backward pass
    loss.backward()
    optimizer.step()

# %%

new_x = Variable(torch.Tensor([[4.0]]))
y_pred = model(new_x)
print("predicted Y value: ", y_pred.data[0][0])
# %%
