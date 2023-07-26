
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.stats import f_oneway
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import metrics
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.svm import SVR
#from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
def Feature_Encoder(X,cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X
data = pd.read_csv("games-classification-dataset.csv")

#data = pd.read_csv('games-regression-dataset.csv')


import re, os, string
import pandas as pd
import os
#fname = "STOPWORD.txt"
#print(os.path.abspath(fname))

# Scikit-learn importings
from sklearn.feature_extraction.text import TfidfVectorizer
def get_stopwords_list(stop_file_path):
    """load stop words """

    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return list(frozenset(stop_set))
def clean_text(text):

    text = text.lower()

    # Removing punctuation
    text = "".join([c for c in text if c not in PUNCTUATION])

    # Removing whitespace and newlines
    text = re.sub('\s+', ' ', text)

    return text
def sort_coo(coo_matrix):
    #Sort a dict with highest score
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    #get the feature names and tf-idf score of top n items

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
        # create a tuples of feature, score
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


def get_keywords(vectorizer, feature_names, doc):
    #Return top k keywords from a doc using TF-IDF method

    # generate tf-idf for the given document
    tf_idf_vector = vectorizer.transform([doc])

    # sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(tf_idf_vector.tocoo())

    # extract only TOP_K_KEYWORDS
    keywords = extract_topn_from_vector(feature_names, sorted_items, TOP_K_KEYWORDS)

    return list(keywords.keys())
#########################
PUNCTUATION = """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
TOP_K_KEYWORDS = 1 # top k number of keywords to retrieve in a ranked document
###########################
data['Description'] = data['Description'].apply(clean_text)
corpora = data['Description'].to_list()
#load a set of stop words
stopwords=get_stopwords_list("stopword.txt")

# Initializing TF-IDF Vectorizer with stopwords
vectorizer = TfidfVectorizer(stop_words=stopwords, smooth_idf=True, use_idf=True)

# Creating vocab with our corpora
# Exlcluding first k docs for testing purpose
vectorizer.fit_transform(corpora[1::])

# Storing vocab
feature_names = vectorizer.get_feature_names()
result = []
for doc in corpora[0:5214]:
    df = {}
    df['full_text'] = doc
    df['top_keywords'] = get_keywords(vectorizer, feature_names, doc)
    result.append(df)

print(result)
data
final = pd.DataFrame(result)
final
ind=data.shape[1]#last colume index
col=final['top_keywords']
lst = col.tolist() # to list
inx=0
#convert each element from list to string
for i in lst:

   st=''.join(i)
   lst[inx]=st
   inx = inx + 1

lst
#insert top_keywords colume in dataframe
data.insert(loc= ind, column='top_keywords', value=lst)
# encoder decoder
cols=('top_keywords',)
Feature_Encoder(data,cols )
data
keyworddata=data
