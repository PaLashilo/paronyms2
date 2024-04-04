import pandas as pd
import numpy as np

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.decomposition import PCA

import pymorphy2
import Levenshtein
import gensim.downloader
import json
import pickle
from os import path

import warnings
warnings.filterwarnings("ignore")


# конфигурация
with open('config.json', 'r') as file:
    config = json.load(file)

data_path = config['data_path']
inference_path = config["bins_directory"]

data = pd.read_csv(data_path, index_col=0)

print("Start installing word2vec")
word2vec_rus = gensim.downloader.load('word2vec-ruscorpora-300')
print("Finish installing word2vec")
word2vec_rus.save(path.join(inference_path, "word2vec_rus.model"))
morph = pymorphy2.MorphAnalyzer()

def get_embedding(word):

    w2v_word = f"{word}_{get_part_of_speech(word)}"

    try:
        emb = word2vec_rus[w2v_word]

    except KeyError:
        return None
    
    return emb


def get_part_of_speech(word):
    
    parsed_word = morph.parse(word)[0]
    pos = parsed_word.tag.POS

    if pos == "ADJF":
        return "ADJ"
    
    return pos 

# add extra columns for embeddings 
for i in range(600):
    data[f"emb_{int(i > 299) + 1}_{i % 300}"] = 0
# add extra columns for Levenshtein distance
data["lev_dist"] = 0

rows_to_drop = []

# add embeddings to dataframe 
for i in range(len(data)):

    # get embs for two words
    word1 = data.word1[i]
    word2 = data.word2[i]
    emb1 = get_embedding(word1)
    emb2 = get_embedding(word2)

    if emb1 is not None and emb2 is not None:
        # add embeddings
        data.loc[i, "emb_1_0":"emb_1_299"] = emb1
        data.loc[i, "emb_2_0":"emb_2_299"] = emb2
        data.loc[i, "lev_dist"] = Levenshtein.distance(word1, word2)
        # print("YES", data.word1[i], data.word2[i])

    else: 
        # delete words that are not in word2vec vocabulary
        rows_to_drop.append(i)
        # print("NO", data.word1[i], data.word2[i])

data = data.drop(rows_to_drop, axis=0).reset_index(drop=True)

# fitting PCA
def fit_pca(X, n):
    pca = PCA(n_components=n)
    pca.fit(X)
    with open(path.join(inference_path, 'pca_model.pkl'), 'wb') as file:
        pickle.dump(pca, file)
    return pca

# transform dataset to n_componets dimansion
def reduce_dimension(pca, X):
    X_transformed = pca.transform(X)
    return X_transformed


X = data.drop(["word1", "word2", "label"], axis=1)
y = data["label"]

pca_n_components = config["pca_n_components"]

# fit PCA
pca = fit_pca(pd.concat([X.loc[:, "emb_1_0":"emb_1_299"], X.loc[:, "emb_2_0":"emb_2_299"].rename(columns={f"emb_2_{i}":f"emb_1_{i}" for i in range(300)}) ], ignore_index=True), pca_n_components)

# reduce dimensuon with PCA
X.loc[:, "emb_1_0":f"emb_1_{pca_n_components-1}"] = reduce_dimension(pca, X.loc[:, "emb_1_0":"emb_1_299"])
X.loc[:, f"emb_2_0":f"emb_2_{pca_n_components-1}"] = reduce_dimension(pca, X.loc[:, "emb_2_0":"emb_2_299"].rename(columns={f"emb_2_{i}":f"emb_1_{i}" for i in range(300)}))

# drop extra columns of embedding
X = X.drop(list(X.columns[pca_n_components:300]) + list(X.columns[300+pca_n_components:-1]), axis=1)

X["lev_dist_2"] = X.lev_dist**2
X["lev_dist_%"] = 0
for i in range(len(data.word1)):
    X["lev_dist_%"][i] = X.lev_dist[i] / max(len(data.word1[i]), len(data.word2[i]))

print("Finish preprocessing")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with open('best_params.json', 'r') as file:
    cat_params = json.load(file)


cat_model = CatBoostClassifier(verbose=0, **cat_params)
cat_model.fit(X_train, y_train)
best_predictions = cat_model.predict(X_test)
best_mae = mean_absolute_error(y_test, best_predictions)
predictions = cat_model.predict_proba(X_test)
print("Finish catboost trining")

treshold = 0.5

mae = mean_absolute_error(y_test, predictions[:, 1])
acc = accuracy_score(y_test, [1 if prob > treshold else 0 for prob in predictions[:, 1]])
print('MAE:', mae, ' Accuracy:', acc)

cat_model.save_model(path.join(inference_path, 'catboost_model.bin'))
print("Save catboost model")
