import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords as sp

import nltk
nltk.download('stopwords')

dataset = pd.read_csv(open(f"datasets/noticias_processadas.tsv"), sep=",", header=0)
dataset = dataset.dropna()

wordcloud = WordCloud(
    stopwords=sp.words('english'),
    background_color="white",
).generate(" ".join(dataset[dataset["fake"] == 1]["text_clean"]))

# mostrar a imagem final
fig, ax = plt.subplots(figsize=(10,6))
ax.imshow(wordcloud, interpolation='bilinear')
ax.set_axis_off()

plt.imshow(wordcloud)
plt.savefig("files/fake_cloud.png")

print("Frequência de palavras no dataset de notícias falsas:")
for word, freq in list(wordcloud.words_.items())[:10]:
    print(f"{word}: {round(freq, 4)}")

# noticias verdadeiras
wordcloud = WordCloud(
    stopwords=sp.words('english'),
    background_color="white",
).generate(" ".join(dataset[dataset["fake"] == 0]["text_clean"]))

# mostrar a imagem final
fig, ax = plt.subplots(figsize=(10,6))
ax.imshow(wordcloud, interpolation='bilinear')
ax.set_axis_off()

plt.imshow(wordcloud)
plt.savefig("files/true_cloud.png")

print("Frequência de palavras no dataset de notícias falsas:")
for word, freq in list(wordcloud.words_.items())[:10]:
    print(f"{word}: {round(freq, 4)}")

