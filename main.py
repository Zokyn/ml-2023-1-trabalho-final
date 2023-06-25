import pandas as pd
import re
from wordcloud import STOPWORDS, WordCloud
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

def average_of_rating(x):
    ### Average of Rating
    # retorna a media da 
    # avaliação (rating)
    return sum(x)/len(x)

def word_count(w):
    ### Word Count
    # faz o processo de 
    # tokeziação da sentença
    # adicionado 1 a cada
    # ocorrencia da palavra
    counts = {}
    words = w.split()
    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 0
    return counts

dataset = pd.read_csv("New_Delhi_reviews.csv")

data = dataset.copy()
print(data.head())
print(data.info())
print(data.isnull().sum())

## REMOVENDO VALORES VAZIOS
data.dropna(inplace=True)
print(data.head())
print(data.info())
print(data.isnull().sum())
