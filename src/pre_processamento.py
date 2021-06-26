import multiprocessing
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords as sp

nltk.download('stopwords')


"""
Pré-processa o texto dos datasets Fake e True e unifica os datasets.
"""

def preprocessing(texto):
    """
    Pré processa o dataset:
    - Remove pontuações
    - Normaliza o texto em letras minúsculas
    - Remove stopwords
    - Remove números
    """

    # remover pontuacoes e normalizar o texto em minúsculo
    texto = re.sub(r'[^\w\s]', '', texto.lower())
    # remover stop words
    texto = ' '.join([word for word in texto.split() if word not in sp.words("english")])
    # remover números
    texto = re.sub(r'[\d]', '', texto)
    # remove whitespaces
    texto = re.sub(r'\s+', ' ', texto).strip()

    return texto 

# carregar noticias
fake = pd.read_csv(open("noticias/Fake.csv"), sep=",", header=0)
true = pd.read_csv(open("noticias/True.csv"), sep=",", header=0)

# atribuir classes
fake.insert(0, "fake", [1]*len(fake))
true.insert(0, "fake", [0]*len(true))

dataset = pd.concat([fake, true], ignore_index=True)

# pre-processar
with multiprocessing.Pool(processes=8) as pool:
    dataset['text_clean'] = pool.map(preprocessing, dataset['text'].tolist())

# salvar
dataset.to_csv(f"datasets/noticias_processadas.tsv", sep=",", index=None)