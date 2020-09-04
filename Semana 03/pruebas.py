#! /bin/python3
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import floresta
stemmer = nltk.stem.RSLPStemmer()
wordwet_lemmatizer = WordNetLemmatizer()
words = ['recomendo', 'gostei', 'péssimo', 'ruim','odiei', 'amei', 'ruim', 'ótimo']
for word in words:
    #lemma2 = wordwet_lemmatizer.lemmatize(word)
    lemma2 = stemmer.stem(word)
    #print(lemma)
    #print("\n")
    #print(lemma2)

import spacy
import pandas as pd
import numpy as np
from unidecode import unidecode
spacyPT = spacy.load('pt_core_news_lg')
b2wCorpus = pd.read_csv("/home/augusto/Documents/GitHub/NLPortugues/Semana 03/data/b2w-10k.csv", encoding='utf-8')
b2wCorpus= b2wCorpus[["review_text", "recommend_to_a_friend"]]
b2wCorpus['recommend_to_a_friend'].replace({'No': 0, 'Yes': 1}, inplace = True)
print(b2wCorpus.head())
for i, row in b2wCorpus.iterrows():
    ifor_val = unidecode(row['review_text']).lower()
    b2wCorpus.at[i,'review_text']= ifor_val
#print(b2wCorpus.head())
#print(spacyPT.Defaults.stop_words)
b2wCorpus['review_text_lema'] = b2wCorpus.review_text.apply(lambda text: 
                                          " ".join(token.lemma_ for token in spacyPT(text) 
                                                   if not token.is_stop and token.pos_ != 'PROPN' ))
print(b2wCorpus.head())

