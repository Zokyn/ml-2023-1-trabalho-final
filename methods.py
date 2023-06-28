import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt


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

def normalization(data: list):
    x_max = data.max()
    x_min = data.min()
    l = []
    for i in range(len(data)):
        x = data.iloc[i]
        x = 2*((x - x_min) / (x_max - x_min))-1

        l.append(x)

    return pd.Series(l)

def average_of_rating(x):
    ### Average of Rating
    # retorna a media da 
    # avaliação (rating)
    return sum(x)/len(x)