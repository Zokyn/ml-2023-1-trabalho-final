import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from methods import normalization, average_of_rating

STOPWORDS = stopwords.words('english')

## 1.LENDO O ARQUIVO DE ENTRADA
dataset = pd.read_csv("New_Delhi_reviews.csv")
data = dataset.copy()

## 2.REMOVENDO VALORES VAZIOS
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

## 3.LIMPANDO OS DOCUMENTOS

# `RATING_REVIEW`` foi feito uma normalizaçào
# para que os dados possuessem apenas valores entre 
# -1 e 1. Atribuindo valores negativos a avaliações
# negativas e valores positivos à positivas a uma 
# nova coluna.

data['rating_norm'] = normalization(data['rating_review'])

# e além disso foi criado uma colun com valores categoricos
# que serão usado como a variavel depeendente a ser predita.
data['rating_class'] = data['rating_review'].replace(
    {1:'Negative',2:'Negative',3:'Neutral',4:'Positive',5:'Positive'})
    
# `REVIEW_FULL` foi retirado prefixos de url
#  de possiveis links no comentario de review, 
#  além disso:
#   1. colocado as palavras em minusculas
#   2. removindo qualquer stopword no texto
#   3. feito o processo de stemming nos tokens
#   4. e transoformado em um novo dataframe
corpus = []
for i in range(len(data)):
    # Documento do corpus acessado
    review = data['review_full'][i]
    
    ## TOKENIZATION
    # remove-se caracteres desncessarios
    review = re.sub(r"[^a-zA-Z0-9\s]", "", review)
    review = re.sub(r"https?://\S+", "", review)
    # divide o documento em palavras uma lista minusculas
    review = review.lower().split()

    ps = PorterStemmer()
    ## STEMMING
    review = [ps.stem(word) for word in review if not word in set(STOPWORDS)]

    review = " ".join(review) 

    # print(review) # DEBUG <-----------------------------------------------
    corpus.append(review)

corpus = pd.DataFrame(corpus, columns=["review_full"])
# Ainda é preciso fazer uma limpeza, porque existe 
# comentários com o mesmo conteudo mas com valores
# de rating diferentes o que pode dificultar o aprendizado
corpus["rating_review"] = data["rating_review"]
corpus_sorted = corpus.groupby(["review_full"]).nunique().sort_values(by='rating_review', ascending=False)
corpus_sorted = corpus_sorted[corpus_sorted["rating_review"] > 1]["rating_review"] 

for h in range(len(corpus_sorted.index)):
    corpus.loc[corpus["review_full"] == corpus_sorted.index[h], 
               "rating_review"] = round(average_of_rating(corpus.loc[corpus["review_full"] == corpus_sorted.index[h], "rating_review"]))

# agora temos os features organizados e separados 
# por documentos prontos para servirem de entrada
data["stem_review"] = corpus["review_full"]

## 4.SEPARANDO OS DADOS
# é preciso separar os conjuntos 
# de treino e teste dos dados e
# utilizar um vectorizer para a 
# frequencia de palavras no review
vectorizer = CountVectorizer(max_features=1500)

x = vectorizer.fit_transform(data["stem_review"].values)
y = data["rating_review"].replace({1: 'negative', 2: 'negative', 3: 'neutral', 4: 'positive', 5: 'positive'}).values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=1)

x_train = x_train.toarray()
x_test = x_test.toarray()

## 5.CONSTRUINDO UM CLASSIFICADOR
# usando Naive Bayes
method = GaussianNB()
method.fit(x_train, y_train)
y_pred = method.predict(x_test)

# usando RandomForest
method = RandomForestClassifier()
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
fscore = metrics.f1_score(y_test, y_pred)
fscore2 = metrics.f1_score(y_test, y_pred2)
fscore3 = metrics.f1_score(y_test, y_pred3)
recall = metrics.f1_score(y_test, y_pred)
recall2 = metrics.f1_score(y_test, y_pred2)
recall3 = metrics.f1_score(y_test, y_pred3)


print(f"[NAIVE BAYES] Accuracy: {accuracy}")
# print(f"[NAIVE BAYES] {fscore}")
# print(f"[NAIVE BAYES] {recall}")
print(f"[RANDOM FOREST] Accuracy: {accuracy2}")
# print(f"[RANDOM FOREST] {fscore2}")
# print(f"[RANDOM FOREST] {recall}")
print(f"[REGRESSÃO] Accuracy: {accuracy3}")
# print(f"[REGRESSÃO] {fscore3}")
# print(f"[REGRESSÃO] {recall}")
# print(data)
# print_dataset(data)
# print_cloudword(data)
