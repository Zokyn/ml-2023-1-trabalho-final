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

def print_dataset(df):
    print(df.head())
    print(df.info())
    print(df.isna().sum())

def print_cloudword(df):
    
    ## CONTANDO E VISUALIZANDO PALAVRAS
    # é criado uma nova coluna com o objeto de 
    # contador de palavras, funcionando como 
    # a tokenização dos elementos, transformando
    # as palavras do documento (nesse caso o review)
    # em pares chave-valor com a frequencia do token

    df["word_count"] = df["review_full"].apply(word_count)
    frequencies = df["review_full"].str.split(expand=True).stack().value_counts()

    wordcloud = WordCloud(width=1000, height=500).generate_from_frequencies(frequencies)

    plt.imshow(wordcloud)
    plt.title("Todas as palavras no review")
    plt.show()
dataset = pd.read_csv("New_Delhi_reviews.csv")

data = dataset.copy()

## REMOVENDO VALORES VAZIOS
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

## LIMPANDO OS DOCUMENTOS
# foi retirado prefixos de url de possiveis links
# no comentario de review, além disso:
#   1. colocado as palavras em minusculas
#   2. removindo qualquer stopword no texto
#   3. transoformado em um novo dataframe
corpus = []
for i in range(len(data)):
    review = data['review_full'][i]
    review = re.sub(r"https\S+", "", review)
    review = re.sub(r"http\S+", "", review)
    review = review.lower()
    review = review.split()

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review if not word in set(STOPWORDS)]

    review = " ".join(review) 

    corpus.append(review)

corpus = pd.DataFrame(corpus, columns=["review"])
corpus["rating"] = data["rating_review"]
print(corpus)
# print_dataset(data)
# print_cloudword(data)
