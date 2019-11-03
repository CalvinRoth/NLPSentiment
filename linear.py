from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import pandas as pd
import numpy as np
import os
import re
from time import time


class Algorithms(object):
    def __init__(self, name, setter, fit=lambda a,x,y: a.fit(x,y), score=lambda a,x,y: a.score(x,y)):
        self.name = name
        self.setter = setter
        self.algo = self.setter()
        self.fit = lambda x,y: fit(self.algo,x,y)
        self.score = lambda x,y: score(self.algo,x,y)

    def reset(self):
        self.algo = self.setter()



# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
    data = {}
    data["review"] = []
    data["score"] = []
    for file_path in os.listdir(directory):
        with open(os.path.join(directory, file_path), "r") as f:
            data["review"].append(f.read())
            #This regex will from the name of a file like '250_1.txt' extract 1(numbers after _)
            data["score"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

# Process the dataset files.
def load_datasets(force_download=False):
    train_df = load_dataset(os.path.join("aclImdb","train"))
    test_df = load_dataset(os.path.join("aclImdb","test"))

    return train_df, test_df

def build_vocabulary():
    return open(os.path.join("aclImdb", "imdb.vocab"), "r").readlines()


def test_algorithms(params, algorithms, score="polarity"):
    train_df, test_df = load_datasets()

    count_vector = CountVectorizer(**params)
    transformer = TfidfTransformer()

    X_train_count = count_vector.fit_transform(train_df["review"])
    X_test_count = count_vector.transform(test_df["review"])

    X_train_tfidf = transformer.fit_transform(X_train_count)
    X_test_tfidf = transformer.transform(X_test_count)

    form_string = "{:6} {:06.2f}\n\t{:5}: {:.4f}\n\t{:5}: {:.4f}\n"

    for a in algorithms:
        start = time()
        a.fit(X_train_tfidf, train_df[score])
        train_score = a.score(X_train_tfidf, train_df[score])
        test_score = a.score(X_test_tfidf, test_df[score])
        taken = time()-start
        print(form_string.format(a.name, taken, "train", train_score, "test", test_score))

knn = Algorithms(
    "KNN",
    lambda : KNeighborsClassifier()
)
mnb = Algorithms(
    "MNB",
    lambda : MultinomialNB(),
    lambda a,x,y: a.fit(x,y),
    lambda a,x,y: np.mean(a.predict(x) == y)
)
lgrg = Algorithms(
    "LogReg",
    lambda : LogisticRegression()
)
dtree = Algorithms(
    "DTree",
    lambda : DecisionTreeClassifier()
)
#svm = Algorithms(
#    "SVM",
#    lambda : SVC()
#) # 35 minutes
linreg = Algorithms(
    "LinReg",
    lambda : LinearRegression()
)


def test_1():
    for i in range(3):
        print("ngram:", i+1)
        test_algorithms({"ngram_range": (1,i+1)}, [knn, mnb, lgrg, dtree, linreg])

def test_2():
    test_algorithms({"stop_words": "english", "ngram_range": (1,2)}, [knn, mnb, lgrg, dtree, linreg])


test_2()
