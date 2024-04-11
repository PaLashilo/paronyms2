# -*- coding: utf-8 -*-
"""inference.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-LqElgwUYlUSCHmLgdalDREtKD5qXXED
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import pickle
import pymorphy2
import Levenshtein
import gensim.downloader
from os import path

inference_path = "inference_binaries"
pca_n_components = 150

cat_model = CatBoostClassifier()
cat_model.load_model(path.join(inference_path, 'catboost_model.bin'))

morph = pymorphy2.MorphAnalyzer()

word2vec_rus = gensim.models.KeyedVectors.load(path.join(inference_path, "word2vec_rus.model"))

def reduce_dimension(X):
    with open(path.join(inference_path, 'pca_model.pkl'), 'rb') as file:
        pca = pickle.load(file)
    X_transformed = pca.transform(X)
    return X_transformed

# get a part of speech needed to make an embedding of word
def get_part_of_speech(word):

    parsed_word = morph.parse(word)[0]
    pos = parsed_word.tag.POS

    if pos == "ADJF":
        return "ADJ"

    return pos

# making embedding by pretrained word2vec model
def get_embedding(word):

    w2v_word = f"{word}_{get_part_of_speech(word)}"

    try:
        emb = word2vec_rus[w2v_word]

    except KeyError:
        return None

    return emb

# function for testing new words
def predict(word1, word2):

    # getting embs and lev dist
    emb1, emb2 = get_embedding(word1), get_embedding(word2)
    if emb1 is None or emb2 is None:
        return "Для слов не нашелся эмбединг"
    
    pca_emb1 = reduce_dimension(emb1.reshape(1, -1))
    pca_emb2 = reduce_dimension(emb2.reshape(1, -1))
    lev_dist = Levenshtein.distance(word1, word2)

    # creating dataframe
    row = pca_emb1.tolist()[0] + pca_emb2.tolist()[0] + [lev_dist] + [lev_dist**2]
    X_new = pd.DataFrame([row], columns=[f"emb_{int(i > pca_n_components - 1) + 1}_{i % pca_n_components}" for i in range(pca_n_components*2)] + ["lev_dist", "lev_dist_2"])

    X_new["lev_dist_%"] = lev_dist / X_new.apply(lambda row: max(len(word1), len(word2)), axis=1)

    prediction = cat_model.predict_proba(X_new)
    res = np.argmax(prediction[0])
    proba = max(prediction[0])

    # result
    return f"Слова {word1} и {word2} {'не '*(not res)}являются паронимами с вероятностью {proba}"


if __name__ == '__main__':
    # не паронимы
    word1 = "приветливый"
    word2 = "страна"

    # паронимы
    # word1 = "целый"
    # word2 = "цельный"

    # тестирование
    print(predict(word1, word2))
