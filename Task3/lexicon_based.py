# Julia Gygax
# jgygax@ethz.ch
# 16-922-064


# %%
import os
import random
import csv
from sklearn.metrics import accuracy_score, f1_score

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
with open('opinion-lexicon-English/negative-words.txt', "r") as f:
    neg = f.read().split()
# %%
with open('opinion-lexicon-English/positive-words.txt', "r") as f:
    pos = f.read().split()

# %%
true_labels = []
estim_labels = []
for path in test_paths:
    with open(path, "r") as f:
        text = f.read().split()

    pos_count = 0
    neg_count = 0
    for word in text:
        if word in pos:
            pos_count += 1
        elif word in neg:
            neg_count += 1

    if 'pos' in path:
        true_labels.append(1)
    else:
        true_labels.append(0)
    if pos_count > neg_count:
        estim_labels.append(1)
    else:
        estim_labels.append(0)

# %%
print("====================================")
print("   USING LEXICON-BASED CLASSIFIER   ")
print("====================================")

print("Accuracy: \t", accuracy_score(true_labels, estim_labels))
print("F-score: \t", f1_score(true_labels, estim_labels))

# ====================================
#    USING LEXICON-BASED CLASSIFIER   
# ====================================
# Accuracy: 	 0.73
# F-score: 	 0.7244897959183672

# %%
