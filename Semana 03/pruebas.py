#! /bin/python3
import pandas as pd
import numpy as np
from unidecode import unidecode
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

b2wCorpus = pd.read_csv("/home/augusto/Documents/GitHub/NLPortugues/Semana 03/data/b2w-10k.csv", encoding='utf-8',nrows=2000)
#b2wCorpus = pd.read_csv("C:\\tmp\\b2w-10k.csv", encoding='utf-8',nrows=100)
b2wCorpus= b2wCorpus[["review_text", "recommend_to_a_friend"]]
b2wCorpus['recommend_to_a_friend'].replace({'No': 0, 'Yes': 1}, inplace = True)

for i, row in b2wCorpus.iterrows():
    ifor_val = unidecode(row['review_text']).lower()
    b2wCorpus.at[i,'review_text']= ifor_val

x = b2wCorpus[['review_text']]
y = b2wCorpus[['recommend_to_a_friend']] 

print(b2wCorpus.recommend_to_a_friend.value_counts()/ b2wCorpus.shape[0])

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=RANDOM_SEED)

text_dataset = tf.data.Dataset.from_tensor_slices(x_train.to_numpy())
vectorize_layer = TextVectorization(
                                        max_tokens=5000,
                                        standardize='lower_and_strip_punctuation',
                                        split='whitespace',
                                        output_mode='int',
                                        pad_to_max_tokens=True
                                        )

vectorize_layer.adapt(text_dataset.batch(64))
vocab_size = len(vectorize_layer.get_vocabulary())

#sys.exit()

model = tf.keras.Sequential([
    ############ Seu código aqui##################
    tf.keras.Input(shape=(1, ),
                   dtype=tf.string),
    vectorize_layer,
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
with tf.device("gpu:0"):
   model.fit(x_train,  y_train, epochs=5, batch_size=10,validation_data=(x_val, y_val))    
   score = model.evaluate(x_val, y_val, verbose=2)
   print(score)
   x_v = x_val.to_numpy()
   x_v = np.append(x_v,['eu odei o produto'])
   x_v = np.append(x_v,['eu gostei o produto'])
   x_v = np.append(x_v,['náo comparei de novo'])
   print(x_v)
   pred = model.predict(x_v)
   print(np.round(pred))

