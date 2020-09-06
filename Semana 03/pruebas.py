#! /bin/python3
#import nltk
#from nltk.stem import WordNetLemmatizer
#from nltk.corpus import floresta
##stemmer = nltk.stem.RSLPStemmer()
#wordwet_lemmatizer = WordNetLemmatizer()
#words = ['recomendo', 'gostei', 'péssimo', 'ruim','odiei', 'amei', 'ruim', 'ótimo']
#for word in words:
    #lemma2 = wordwet_lemmatizer.lemmatize(word)
 #   lemma2 = stemmer.stem(word)
    #print(lemma)
    #print("\n")
    #print(lemma2)

import spacy
import pandas as pd
import numpy as np
from unidecode import unidecode
import re
import sys
spacyPT = spacy.load('pt_core_news_lg')
#b2wCorpus = pd.read_csv("/home/augusto/Documents/GitHub/NLPortugues/Semana 03/data/b2w-10k.csv", encoding='utf-8',nrows=2000)
b2wCorpus = pd.read_csv("C:\\tmp\\b2w-10k.csv", encoding='utf-8',nrows=20)
b2wCorpus= b2wCorpus[["review_text", "recommend_to_a_friend"]]
b2wCorpus['recommend_to_a_friend'].replace({'No': 0, 'Yes': 1}, inplace = True)

for i, row in b2wCorpus.iterrows():
    ifor_val = unidecode(row['review_text']).lower()
    b2wCorpus.at[i,'review_text']= ifor_val
#print(b2wCorpus.head())
#print(spacyPT.Defaults.stop_words)
b2wCorpus['review_text_lema'] = b2wCorpus.review_text.apply(lambda text: 
                                          " ".join(re.sub('[^A-Za-z0-9]+', ' ', token.lemma_) for token in spacyPT(text) 
                                                   if not token.is_stop and token.pos_ not in ['PROPN','NUM','SYM'] ))
for i, row in b2wCorpus.iterrows():
    ifor_val =   ' '.join((row['review_text_lema']).split())
    b2wCorpus.at[i,'review_text_lema']= ifor_val


import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


X = b2wCorpus[['review_text']]
y = b2wCorpus[['recommend_to_a_friend']] 

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

textVecLayer = TextVectorization( output_mode="int", pad_to_max_tokens=True)
textVecLayer.adapt(X.to_numpy())
vocab_size = len(textVecLayer.get_vocabulary())

#sys.exit()

model = tf.keras.Sequential([
    ############ Seu código aqui##################
    tf.keras.Input(shape=(1, ),
                   dtype=tf.string),
    textVecLayer,
    tf.keras.layers.Embedding(input_dim=vocab_size,
                              output_dim = 32),
    ##############################################
    # Conv1D + global max pooling
    layers.Conv1D(128, 7, padding='valid', activation='relu', strides=3),
    layers.Conv1D(128, 7, padding='valid', activation='relu', strides=3),
    layers.GlobalMaxPooling1D(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid'),
])

model.compile(
    loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x = x_train, y = y_train, epochs=400, batch_size=10,validation_data=(x_val, y_val))    


