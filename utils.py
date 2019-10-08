import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.text import Tokenizer
import json
import nltk
import string
import re

scalings = ["normal"]

def clean_text(text):
    
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)
    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    ## Stemming
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    
    return text

def scale_data(X, mode = "normal"):
    """ Scale the data
    """
    if mode == "normal":
        return X/(1 + np.sum(X, axis = 1)[:, np.newaxis])
    else:
        return X

def encode_labels(dataframe, unique_labels, label_column):
    labels = np.array(dataframe[label_column])
    y_encoded = np.array([[1 if unique_label == label else 0 for unique_label in unique_labels] for label in labels])
    return y_encoded

def fetch_normalised_data_from_dataframe(dataframe, unique_labels, normalise_columns, label_column, scaling = None, encode = False):
    X = np.array(dataframe[normalise_columns])
    y = np.array(dataframe[label_column])
    scaled_X = None
    encoded_y = None
    if scaling in scalings:
        scaled_X = scale_data(X, mode = scaling)
    else:
        scaled_X = X
    
    keep_columns = list(set(dataframe.columns) - set([label_column]) - set(normalise_columns))

    scaled_X = np.append(scaled_X, np.array(dataframe[keep_columns]), axis = 1)

    if encode == True:
        encoded_y = encode_labels(dataframe, unique_labels, label_column)
    else:
        encoded_y = np.array(dataframe[label_column])

    return (scaled_X, encoded_y)


def preprocess_from_df(dataframe, unique_labels, normalise_columns, label_column, scaling = "normal", encode = True, split = True, test_frac = 0.2):
    X, y = fetch_normalised_data_from_dataframe(dataframe, unique_labels, normalise_columns, label_column = label_column, scaling = scaling, encode = encode)
    if split == True:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_frac)
        return (X_train, X_test, y_train, y_test)
    else:
        return X, y

def tokenizer_from_json(json_string):
    tokenizer_config = json.loads(json_string)
    config = tokenizer_config.get('config')

    word_counts = json.loads(config.pop('word_counts'))
    word_docs = json.loads(config.pop('word_docs'))
    index_docs = json.loads(config.pop('index_docs'))
    # Integer indexing gets converted to strings with json.dumps()
    index_docs = {int(k): v for k, v in index_docs.items()}
    index_word = json.loads(config.pop('index_word'))
    index_word = {int(k): v for k, v in index_word.items()}
    word_index = json.loads(config.pop('word_index'))

    tokenizer = Tokenizer(**config)
    tokenizer.word_counts = word_counts
    tokenizer.word_docs = word_docs
    tokenizer.index_docs = index_docs
    tokenizer.word_index = word_index
    tokenizer.index_word = index_word

    return tokenizer

def save_confusion_matrix(y_pred, y_true, labels, filename):
    mat = confusion_matrix(y_pred, y_true)
    mat = np.round(mat.astype('float') / mat.sum(axis=1)[:, np.newaxis], 2)
    ax= plt.subplot()
    sns.heatmap(mat, annot = True, ax = ax, fmt = 'g', cmap = 'Greens')
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title('Multiclass Classification Results')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.savefig('figures/{0}_confusion_matrix.png'.format(filename), bbox_inches='tight')

if __name__ == "__main__":
    pass