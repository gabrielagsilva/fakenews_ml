import fasttext
import fasttext.util
import multiprocessing
import numpy as np
import pandas as pd

fasttext.util.download_model('en', if_exists='ignore')
model = fasttext.load_model('cc.en.300.bin')

def get_features(texto):
    try:
        palavras = texto.split(' ')
        palavras_features = np.zeros(300)
        for palavra in palavras:
            palavras_features += model.get_word_vector(palavra)
        return palavras_features / len(palavras)
    except:
        print(texto)
        return np.zeros(300)

# carregar noticias
dataset = pd.read_csv(open(f"datasets/noticias_processadas.tsv"), sep=",", header=0)

# extrair features com fasttext
with multiprocessing.Pool(processes=8) as pool:
    features = pool.map(get_features, dataset['text_clean'].tolist())

features =  pd.DataFrame(features)
dataset = pd.concat([dataset, features], axis=1)

# salvar
dataset.to_csv(f"datasets/features.tsv", sep=",", index=None)
