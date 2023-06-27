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

## 1.LENDO O ARQUIVO DE ENTRADA
dataset = pd.read_csv("New_Delhi_reviews.csv")

data = dataset.copy()

## 2.REMOVENDO VALORES VAZIOS
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

## 3.LIMPANDO OS DOCUMENTOS
# foi retirado prefixos de url de possiveis links
# no comentario de review, além disso:
#   1. colocado as palavras em minusculas
#   2. removindo qualquer stopword no texto
#   3. feito o processo de stemming nos tokens
#   4. e transoformado em um novo dataframe
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
    
# *preciso revisar oq ue acontecer aqui* 
corpus = pd.DataFrame(corpus, columns=["review_full"])
corpus["rating_review"] = data["rating_review"]
# corpus_sorted = corpus.groupby(["review_full"]).nunique().sort_values(by='rating_review', ascending=False)
# corpus_sorted = corpus_sorted[corpus_sorted["rating_review"] > 1]["rating_review"]
# k = corpus_sorted.index

# for h in range(len(k)):
#     corpus.loc[corpus["review_full"] == k[h], "rating_review"] = round(average_of_rating(corpus.loc[corpus["review_full"] == k[h], "rating_review"]))
# # *ate aqui mais ou menos* 

data["stem_review"] = corpus["review_full"]

## 4.SEPARANDO OS DADOS
# é preciso separar os conjuntos 
# de treino e teste dos dados e
# utilizar um vectorizer para a 
# frequencia de palavras no review

x = data["stem_review"].values
y = data["rating_review"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=1)

vectorizer = CountVectorizer(max_features=1500)
x_train = vectorizer.fit_transform(x_train).toarray()
x_test = vectorizer.transform(x_test).toarray()

## 5.CONSTRUINDO UM CLASSIFICADOR
# usando Naive Bayes
method = GaussianNB()
method.fit(x_train, y_train)
y_pred = method.predict(x_test)

# usando RandomForest
method = RandomForestClassifier(n_estimators=1000)
method.fit(x_train, y_train)
y_pred2 = method.predict(x_test) 

# Usando Regressão
method = LogisticRegression(max_iter=1000)
method.fit(x_train, y_train)
y_pred3 = method.predict(x_test)

## 6.ACESSANDO OS RESULTADOS
accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy2 = metrics.accuracy_score(y_test, y_pred2)
accuracy3 = metrics.accuracy_score(y_test, y_pred3)

print(f"[NAIVE BAYES] Accuracy: {accuracy}")
print(f"[RANDOM FOREST] Accuracy: {accuracy2}")
print(f"[REGRESSÃO] Accuracy: {accuracy3}")

# print(data)
# print_dataset(data)
# print_cloudword(data)
