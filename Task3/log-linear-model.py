# Julia Gygax
# jgygax@ethz.ch
# 16-922-064

# %%
import heapq
import nltk
import numpy as np
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
rootdir = 'txt_sentoken'
samples_paths = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        name = os.path.join(subdir, file)
        samples_paths.append(name)
# %%
random.seed(1)
random.shuffle(samples_paths)
test_paths = samples_paths[:400]
train_paths = samples_paths[400:]
# %%

article_text = ''
texts = []
labels = []

for path in samples_paths:
    with open(path, "r") as f:
        text = f.read()
        article_text += text
        texts.append(text)
    if 'pos' in path:
        labels.append(1)
    else:
        labels.append(0)

# %%
for i in range(len(texts)):
    texts[i] = texts[i].lower()
    texts[i] = re.sub(r'\W', ' ', texts[i])
    texts[i] = re.sub(r'\s+', ' ', texts[i])

# %%
wordfreq = {}
for review in texts:
    tokens = nltk.word_tokenize(review)
    for token in tokens:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1
# %%
most_freq = heapq.nlargest(8783, wordfreq, key=wordfreq.get)
# %%
review_vectors = []
for review in texts:
    review_tokens = nltk.word_tokenize(review)
    review_vec = []
    for token in most_freq:
        if token in review_tokens:
            review_vec.append(1)
        else:
            review_vec.append(0)
    review_vectors.append(review_vec)
# %%
review_vectors = np.asarray(review_vectors)

with open('onehot10.npy', 'wb') as f:
    np.save(f, review_vectors)
    np.save(f, np.asarray(labels))
# %%
pca = PCA(n_components=200)
data = pca.fit_transform(review_vectors)

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
