# General libraries
import re
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


def feature_encoder(x, columns):
    for c in columns:
        lbl = LabelEncoder()
        lbl.fit(list(x[c].values))
        x[c] = lbl.transform(list(x[c].values))
    return x


data = pd.read_csv('E:/games-regression-dataset.csv')

# understand the data
print(data.columns)
print(data.head(5))
print(data.info())
print(data.describe())
print(data.isnull().sum())
print(data.shape)

name = "STOPWORD.txt"
print(os.path.abspath(name))


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
    # Sort a dict with the highest score
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_name, sorted_items, topn=10):
    # get the feature names and tf-idf score of top n items

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_name[idx])
        # create a tuples of feature, score
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


def get_keywords(vectorized, feature_name, docu):
    # Return top k keywords from a doc using TF-IDF method

    # generate tf-idf for the given document
    tf_idf_vector = vectorized.transform([docu])

    # sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(tf_idf_vector.tocoo())

    # extract only TOP_K_KEYWORDS
    keywords = extract_topn_from_vector(feature_name, sorted_items, TOP_K_KEYWORDS)

    return list(keywords.keys())


#########################
PUNCTUATION = """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
TOP_K_KEYWORDS = 1  # top k number of keywords to retrieve in a ranked document
###########################
data['Description'] = data['Description'].apply(clean_text)
corpora = data['Description'].to_list()
# load a set of stop words
stopwords = get_stopwords_list("stopword.txt")

# Initializing TF-IDF vectorized with stopwords
vectorize = TfidfVectorizer(stop_words=stopwords, smooth_idf=True, use_idf=True)

# Creating vocab with our corpora
# Excluding first 10 docs for testing purpose
vectorize.fit_transform(corpora[1::])

# Storing vocab
feature_names = vectorize.get_feature_names_out()
result = []
for doc in corpora[0:5214]:
    df = {}
    df['full_text'] = doc
    df['top_keywords'] = get_keywords(vectorize, feature_names, doc)
    result.append(df)

print(result)
final = pd.DataFrame(result)
# uni = []
ind = data.shape[1]
col = final['top_keywords']
lst = col.tolist()  # data is pandas series
inx = 0
for i in lst:
    st = ''.join(i)
    lst[inx] = st
    inx = inx + 1

data.insert(loc=ind, column='top_keywords', value=lst)

cols = ('top_keywords',)
feature_encoder(data, cols)
key_word_data = data
