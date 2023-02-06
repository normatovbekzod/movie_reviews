#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import glob
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# In[2]:


# Downloading the file

nltk.download('movie_reviews', quiet=True)

# Loading the reviews (old fucnction)

def load_reviews(dirname):
    res = {'filename': [], 'kind': [], 'text': []}
    files = glob.glob(f"{dirname}/*/*.txt")
    for f in files:
        fd = open(f)
        res['text'].append(fd.read())
        fd.close()
        elems = f.split('/')
        res['kind'].append(elems[-2])
        res['filename'].append(elems[-1])
    return pd.DataFrame(res)

dataset = load_reviews(str(Path.home()) + '/nltk_data/corpora/movie_reviews')


# In[3]:


# Target values: mapping 'neg' and 'pos' to 0 and 1

y = []
df = dataset.copy()
for index, row in df.iterrows():
    if row['kind'] == 'neg':
        y.append(0)
    elif row['kind'] == 'pos':
        y.append(1)


# In[4]:


# Split into training and test sets

X = df['text']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Instantiate Vectorizer

cv = CountVectorizer(stop_words='english')
docmtr = cv.fit_transform(X_train)

# Training

clf = MultinomialNB().fit(docmtr, y_train)


# In[ ]:


# final prediction function

def predict_sentiment(text):
    review = [text]
    pred = int(clf.predict(cv.transform(review)))
    if pred == 0:
        return 'neg'
    elif pred == 1:
        return 'pos'

