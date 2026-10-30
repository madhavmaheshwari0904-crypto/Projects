from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

base_dir = Path(__file__).resolve().parent.parent
dataset_path = base_dir / "qoute_dataset .csv"
df = pd.read_csv(dataset_path)

#print(df.head())
#print(df.shape)
"""                                              quote           Author
0  “The world as we have created it is a process ...  Albert Einstein
1  “It is our choices, Harry, that show what we t...     J.K. Rowling
2  “There are only two ways to live your life. On...  Albert Einstein
3  “The person, be it gentleman or lady, who has ...      Jane Austen
4  “Imperfection is beauty, madness is genius and...   Marilyn Monroe
(3038, 2)
"""
quotes = df['quote']
#print(quotes.head())
""" 0    “The world as we have created it is a process ...
1    “It is our choices, Harry, that show what we t...
2    “There are only two ways to live your life. On...
3    “The person, be it gentleman or lady, who has ...
4    “Imperfection is beauty, madness is genius and...
Name: quote, dtype: str
"""
from tensorflow.keras.preprocessing.text import Tokenizer
vocab_size = 10000
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(quotes)
word_index = tokenizer.word_index
#print(len(word_index))8213
#print(list(word_index.items())[:10])
""""[('the', 1), ('you', 2),
('to', 3), ('and', 4),
('a', 5), ('i', 6),
('is', 7), ('of', 8), 
('that', 9), ('it', 10)]
"""
sequence = tokenizer.texts_to_sequences(quotes)
for i in range(3):
    #print(quotes[i])
    #print(sequence[i])
    """“The world as we have created it is a process of our thinking. It cannot be changed without changing our thinking.”
[732, 61, 30, 19, 17, 901, 10, 7, 5, 1188, 8, 71, 290, 10, 149, 12, 835, 103, 776, 71, 290, 136]
“It is our choices, Harry, that show what we truly are, far more than our abilities.”
[977, 7, 71, 902, 388, 9, 431, 21, 19, 479, 14, 299, 51, 55, 71, 2545, 136]
“There are only two ways to live your life. One is as though nothing is a miracle. The other is as though everything is a miracle.”
[1384, 14, 53, 203, 700, 3, 80, 15, 37, 36, 7, 30, 300, 92, 7, 5, 1073, 1, 104, 7, 30, 300, 126, 7, 5, 1073, 136]
    """
X = []
y = []

for seq in sequence:
  for i in range(1,len(seq)):
    input_seq = seq[:i]
    output_seq = seq[i]
    X.append(input_seq)
    y.append(output_seq)
#print(len(X),len(y))86549 86549
max_len = max(len(x) for x in X)
#print(max_len)748
from tensorflow.keras.preprocessing.sequence import pad_sequences
X_padded = pad_sequences(X, maxlen=max_len, padding='pre')
y = np.array(y)
#print(X_padded.shape)(86549, 748)
from tensorflow.keras.utils import to_categorical
y_one_hot = to_categorical(y, num_classes=vocab_size)
#print(y.shape)(86549,)
#print(y_one_hot.shape)(86549, 10000)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,LSTM, Dense
embedding_dim = 50
rnn_units = 128
rnn_model = Sequential()

rnn_model.add(
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)
)
rnn_model.add(SimpleRNN(units=rnn_units))
rnn_model.add(Dense(units=vocab_size, activation='softmax'))
rnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
rnn_model.summary()
"""Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ embedding (Embedding)                │ ?                           │     0 (unbuilt) │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ simple_rnn (SimpleRNN)               │ ?                           │     0 (unbuilt) │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ ?                           │     0 (unbuilt) │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 0 (0.00 B)
 Trainable params: 0 (0.00 B)
 Non-trainable params: 0 (0.00 B)
"""
lstm_model = Sequential()
lstm_model.add(
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)
)
lstm_model.add(LSTM(units=rnn_units))
lstm_model.add(Dense(units=vocab_size, activation='softmax'))
lstm_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
lstm_model.summary()
"""Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ embedding (Embedding)                │ ?                           │     0 (unbuilt) │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ simple_rnn (SimpleRNN)               │ ?                           │     0 (unbuilt) │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ ?                           │     0 (unbuilt) │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 0 (0.00 B)
 Trainable params: 0 (0.00 B)
 Non-trainable params: 0 (0.00 B)
Model: "sequential_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ embedding_1 (Embedding)              │ ?                           │     0 (unbuilt) │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm (LSTM)                          │ ?                           │     0 (unbuilt) │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ ?                           │     0 (unbuilt) │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 0 (0.00 B)
 Trainable params: 0 (0.00 B)
 Non-trainable params: 0 (0.00 B)
"""
from tensorflow.keras.models import load_model

lstm_model = load_model(base_dir / "lstm_model (1).h5")
index_to_word = {}
for word, index in word_index.items():
  index_to_word[index] = word

from tensorflow.keras.preprocessing.sequence import pad_sequences
def predictor(model,tokenizer,text,max_len):
    text = text.lower()

    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=max_len, padding='pre')

    pred = model.predict(seq,verbose = 0)
    pred_index = np.argmax(pred)
    return index_to_word.get(pred_index, "")
seed_text = "what are you"
next_word = predictor(lstm_model,tokenizer,seed_text,max_len)
print(next_word)
def generate_text(model,tokenizer,seed_text,max_len,n_words):
    for _ in range(n_words):
        next_word = predictor(model,tokenizer,seed_text,max_len)
        if next_word == "":
            break
        seed_text += " " + next_word
    return seed_text
seed = "are you a "
generate_text = generate_text(lstm_model,tokenizer,seed,max_len,10)
#print(generate_text)
#are you a  yours place i step the mean if it through not
import pickle
with open("tokenizer.pkl", "wb") as f:
  pickle.dump(tokenizer, f)
with open("max_len.pkl", "wb") as f:
  pickle.dump(max_len, f)
     